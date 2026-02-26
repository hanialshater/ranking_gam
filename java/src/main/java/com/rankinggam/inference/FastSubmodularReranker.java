package com.rankinggam.inference;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Optimized submodular reranker: allocation-free heap, incremental groupwise state.
 *
 * <p>Improvements over {@link SubmodularGamReranker}:
 * <ul>
 *   <li><b>Array-based max-heap</b> — no Candidate objects, no GC pressure.
 *       Uses parallel double[]/int[] arrays for gain and index, with inline
 *       sift-up/sift-down.
 *   <li><b>Precomputed base scores</b> — computed once and reused. Accepts
 *       external scores or computes via the model.
 *   <li><b>Incremental novelty state</b> — for category/brand novelty features,
 *       maintains a running count map per category. Computing novelty for a
 *       candidate is O(numTowers) instead of O(|S| * numTowers).
 *   <li><b>Fixed-size selected array</b> — no per-step array allocation.
 * </ul>
 *
 * <p>For novelty-only diversity (the common case), this reduces per-candidate
 * evaluation from O(|S|) to O(1), making the greedy loop O(k * evals) instead
 * of O(k * evals * |S|).
 */
public final class FastSubmodularReranker {

    private final DistilledGamModel baseModel;
    private final ConcavePwlFunction[] diversityTowers;
    private final double maxDiversityScore;

    // Novelty tower config: for each tower, which feature column to check
    private final int numTowers;
    private final int[] noveltyColumns;   // -1 if tower is not novelty-type

    /**
     * @param baseModel       base GAM scoring model
     * @param diversityTowers concave PWL diversity towers
     * @param noveltyColumns  per-tower: feature column index for novelty, or -1 if not novelty.
     *                        Length must equal diversityTowers.length.
     */
    public FastSubmodularReranker(DistilledGamModel baseModel,
                                   ConcavePwlFunction[] diversityTowers,
                                   int[] noveltyColumns) {
        this.baseModel = baseModel;
        this.diversityTowers = diversityTowers;
        this.numTowers = diversityTowers.length;
        this.noveltyColumns = noveltyColumns.clone();

        double maxDiv = 0;
        for (ConcavePwlFunction tower : diversityTowers) {
            maxDiv += tower.evaluateAtMax();
        }
        this.maxDiversityScore = maxDiv;
    }

    /**
     * Rerank with precomputed base scores.
     *
     * @param features       [numDocs][numFeatures] feature matrix
     * @param baseScores     precomputed base scores (length numDocs)
     * @param k              number of items to select
     * @param maxEvalsPerPos max re-evaluations per position (0 = unlimited)
     * @return selected indices in selection order
     */
    public int[] rerank(double[][] features, double[] baseScores, int k, int maxEvalsPerPos) {
        final int n = features.length;
        if (k <= 0 || n == 0) return new int[0];
        k = Math.min(k, n);
        if (maxEvalsPerPos <= 0) maxEvalsPerPos = n;

        // ── Array-based max-heap ──
        // heapGain[i] = gain value, heapIdx[i] = document index, heapStep[i] = computedAtStep
        double[] heapGain = new double[n];
        int[] heapIdx = new int[n];
        int[] heapStep = new int[n];
        int heapSize = 0;

        for (int i = 0; i < n; i++) {
            heapGain[heapSize] = baseScores[i] + maxDiversityScore;
            heapIdx[heapSize] = i;
            heapStep[heapSize] = -1;
            heapSize++;
            siftUp(heapGain, heapIdx, heapStep, heapSize - 1);
        }

        // ── Incremental novelty state ──
        // For each novelty tower, map from category value -> count in selected set
        @SuppressWarnings("unchecked")
        Map<Long, Integer>[] noveltyCounts = new HashMap[numTowers];
        for (int t = 0; t < numTowers; t++) {
            if (noveltyColumns[t] >= 0) {
                noveltyCounts[t] = new HashMap<>();
            }
        }

        int[] selected = new int[k];
        boolean[] inSelected = new boolean[n];
        int selectedCount = 0;

        for (int step = 0; step < k; step++) {
            int evals = 0;
            int bestHeapPos = -1;

            while (heapSize > 0) {
                // Pop top
                int topIdx = heapIdx[0];
                double topGain = heapGain[0];
                int topStep = heapStep[0];

                // Remove from heap
                heapSize--;
                if (heapSize > 0) {
                    heapGain[0] = heapGain[heapSize];
                    heapIdx[0] = heapIdx[heapSize];
                    heapStep[0] = heapStep[heapSize];
                    siftDown(heapGain, heapIdx, heapStep, heapSize, 0);
                }

                if (inSelected[topIdx]) continue;

                if (topStep == step) {
                    // True best — put it back temporarily so we can find it
                    // Actually we already popped it, just record it
                    selected[step] = topIdx;
                    bestHeapPos = 0; // sentinel: found
                    // Update incremental state
                    inSelected[topIdx] = true;
                    selectedCount++;
                    for (int t = 0; t < numTowers; t++) {
                        if (noveltyColumns[t] >= 0) {
                            long cat = Double.doubleToRawLongBits(features[topIdx][noveltyColumns[t]]);
                            noveltyCounts[t].merge(cat, 1, Integer::sum);
                        }
                    }
                    break;
                }

                // Recompute diversity for this candidate
                double divScore = computeDiversity(features, topIdx, noveltyCounts, selectedCount);
                double newGain = baseScores[topIdx] + divScore;
                evals++;

                if (evals >= maxEvalsPerPos) {
                    // Budget exhausted: this candidate is our best so far
                    // Check if any already-evaluated-this-step items in heap are better
                    double bestGain = newGain;
                    int bestIdx = topIdx;

                    // Scan heap for items evaluated this step (they're near the top)
                    for (int h = 0; h < heapSize; h++) {
                        if (heapStep[h] == step && !inSelected[heapIdx[h]] && heapGain[h] > bestGain) {
                            // Swap: put our current best back, take this one
                            // Push current best into heap
                            heapGain[heapSize] = bestGain;
                            heapIdx[heapSize] = bestIdx;
                            heapStep[heapSize] = step;
                            heapSize++;
                            siftUp(heapGain, heapIdx, heapStep, heapSize - 1);
                            // Take the better one (remove from heap)
                            bestGain = heapGain[h];
                            bestIdx = heapIdx[h];
                            // Remove h from heap
                            heapSize--;
                            if (h < heapSize) {
                                heapGain[h] = heapGain[heapSize];
                                heapIdx[h] = heapIdx[heapSize];
                                heapStep[h] = heapStep[heapSize];
                                siftDown(heapGain, heapIdx, heapStep, heapSize, h);
                                h--; // re-check this position
                            }
                        }
                    }
                    if (bestIdx != topIdx) {
                        // Push topIdx back with its evaluated gain
                        heapGain[heapSize] = newGain;
                        heapIdx[heapSize] = topIdx;
                        heapStep[heapSize] = step;
                        heapSize++;
                        siftUp(heapGain, heapIdx, heapStep, heapSize - 1);
                    }

                    selected[step] = bestIdx;
                    bestHeapPos = 0;
                    inSelected[bestIdx] = true;
                    selectedCount++;
                    for (int t = 0; t < numTowers; t++) {
                        if (noveltyColumns[t] >= 0) {
                            long cat = Double.doubleToRawLongBits(features[bestIdx][noveltyColumns[t]]);
                            noveltyCounts[t].merge(cat, 1, Integer::sum);
                        }
                    }
                    break;
                }

                // Push back with updated gain
                heapGain[heapSize] = newGain;
                heapIdx[heapSize] = topIdx;
                heapStep[heapSize] = step;
                heapSize++;
                siftUp(heapGain, heapIdx, heapStep, heapSize - 1);
            }

            if (bestHeapPos < 0) break; // shouldn't happen
        }

        return selected;
    }

    /**
     * Rerank computing base scores internally.
     */
    public int[] rerank(double[][] features, int k, int maxEvalsPerPos) {
        return rerank(features, baseModel.score(features), k, maxEvalsPerPos);
    }

    /**
     * Rerank with unlimited eval budget.
     */
    public int[] rerank(double[][] features, int k) {
        return rerank(features, k, 0);
    }

    /**
     * Compute diversity score for a candidate using incremental state.
     */
    private double computeDiversity(double[][] features, int candidateIdx,
                                     Map<Long, Integer>[] noveltyCounts, int selectedCount) {
        double divScore = 0;
        for (int t = 0; t < numTowers; t++) {
            double gwFeat;
            if (noveltyColumns[t] >= 0 && noveltyCounts[t] != null) {
                if (selectedCount == 0) {
                    gwFeat = 1.0; // max novelty when S is empty
                } else {
                    long cat = Double.doubleToRawLongBits(features[candidateIdx][noveltyColumns[t]]);
                    int matches = noveltyCounts[t].getOrDefault(cat, 0);
                    gwFeat = 1.0 - (double) matches / selectedCount;
                }
            } else {
                gwFeat = 1.0; // fallback for unsupported tower types
            }
            divScore += diversityTowers[t].evaluate(gwFeat);
        }
        return divScore;
    }

    // ── Array-based max-heap operations ──

    private static void siftUp(double[] gain, int[] idx, int[] step, int pos) {
        while (pos > 0) {
            int parent = (pos - 1) >>> 1;
            if (gain[pos] > gain[parent]) {
                swap(gain, idx, step, pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    private static void siftDown(double[] gain, int[] idx, int[] step, int size, int pos) {
        while (true) {
            int left = 2 * pos + 1;
            int right = left + 1;
            int largest = pos;

            if (left < size && gain[left] > gain[largest]) largest = left;
            if (right < size && gain[right] > gain[largest]) largest = right;

            if (largest != pos) {
                swap(gain, idx, step, pos, largest);
                pos = largest;
            } else {
                break;
            }
        }
    }

    private static void swap(double[] gain, int[] idx, int[] step, int a, int b) {
        double tg = gain[a]; gain[a] = gain[b]; gain[b] = tg;
        int ti = idx[a]; idx[a] = idx[b]; idx[b] = ti;
        int ts = step[a]; step[a] = step[b]; step[b] = ts;
    }
}
