package com.rankinggam.inference;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Optimized submodular reranker: compiled towers, array counters, allocation-free heap.
 *
 * <p>Improvements over {@link SubmodularGamReranker}:
 * <ul>
 *   <li><b>Compiled diversity towers</b> — {@link CompiledConcavePwl} inlines all knot
 *       parameters as JIT constants (flat if/else, no loops for K<=6).
 *   <li><b>Int-array novelty counters</b> — when category values are small non-negative
 *       integers, uses {@code int[maxCat+1]} instead of HashMap for O(1) indexed lookup.
 *       Falls back to HashMap for non-integer or unknown-cardinality features.
 *   <li><b>Array-based max-heap</b> — no Candidate objects, no GC pressure.
 *       Uses parallel double[]/int[] arrays for gain and index, with inline
 *       sift-up/sift-down.
 *   <li><b>Precomputed base scores</b> — computed once and reused. Accepts
 *       external scores or computes via the model.
 *   <li><b>Fixed-size selected array</b> — no per-step array allocation.
 * </ul>
 *
 * <p>For novelty-only diversity (the common case), this reduces per-candidate
 * evaluation from O(|S|) to O(1), making the greedy loop O(k * evals) instead
 * of O(k * evals * |S|).
 */
public final class FastSubmodularReranker {

    private final DistilledGamModel baseModel;
    private final CompiledConcavePwl.ConcaveEval[] compiledTowers;
    private final double maxDiversityScore;

    // Novelty tower config: for each tower, which feature column to check
    private final int numTowers;
    private final int[] noveltyColumns;   // -1 if tower is not novelty-type

    // Int-array counter config: per tower, max category value for array indexing.
    // -1 means use HashMap fallback.
    private final int[] maxCategoryValues;

    /**
     * Construct with compiled diversity towers and optional int-array counter hints.
     *
     * @param baseModel         base GAM scoring model
     * @param diversityTowers   concave PWL diversity towers
     * @param noveltyColumns    per-tower: feature column index for novelty, or -1 if not novelty
     * @param maxCategoryValues per-tower: max integer category value for array counters, or -1 for HashMap fallback
     */
    public FastSubmodularReranker(DistilledGamModel baseModel,
                                   ConcavePwlFunction[] diversityTowers,
                                   int[] noveltyColumns,
                                   int[] maxCategoryValues) {
        this.baseModel = baseModel;
        this.numTowers = diversityTowers.length;
        this.noveltyColumns = noveltyColumns.clone();
        this.maxCategoryValues = maxCategoryValues.clone();

        // Compile towers
        this.compiledTowers = new CompiledConcavePwl.ConcaveEval[numTowers];
        double maxDiv = 0;
        for (int t = 0; t < numTowers; t++) {
            CompiledConcavePwl compiled = CompiledConcavePwl.compile(diversityTowers[t]);
            this.compiledTowers[t] = compiled.evaluator();
            maxDiv += compiled.evaluateAtMax();
        }
        this.maxDiversityScore = maxDiv;
    }

    /**
     * Backward-compatible constructor. Auto-detects integer category values from
     * the first rerank call (uses HashMap until then, then switches to array counters
     * if values are small non-negative integers). For simplicity, always uses HashMap.
     *
     * @param baseModel       base GAM scoring model
     * @param diversityTowers concave PWL diversity towers
     * @param noveltyColumns  per-tower: feature column index for novelty, or -1 if not novelty
     */
    public FastSubmodularReranker(DistilledGamModel baseModel,
                                   ConcavePwlFunction[] diversityTowers,
                                   int[] noveltyColumns) {
        this(baseModel, diversityTowers, noveltyColumns, defaultMaxCats(diversityTowers.length));
    }

    private static int[] defaultMaxCats(int numTowers) {
        int[] maxCats = new int[numTowers];
        Arrays.fill(maxCats, -1); // HashMap fallback
        return maxCats;
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

        // ── Pre-extract category values as int arrays ──
        // For towers using array counters, extract category ints from features once.
        // For HashMap towers, extract long keys once.
        int[][] catInts = new int[numTowers][];   // only for array-counter towers
        long[][] catLongs = new long[numTowers][]; // only for HashMap towers
        for (int t = 0; t < numTowers; t++) {
            if (noveltyColumns[t] < 0) continue;
            int col = noveltyColumns[t];
            if (maxCategoryValues[t] >= 0) {
                // Array counter: extract as int
                catInts[t] = new int[n];
                for (int i = 0; i < n; i++) {
                    catInts[t][i] = (int) features[i][col];
                }
            } else {
                // HashMap: auto-detect if values are small non-negative integers
                boolean allSmallInt = true;
                int maxVal = 0;
                for (int i = 0; i < n; i++) {
                    double v = features[i][col];
                    int iv = (int) v;
                    if (v != iv || iv < 0 || iv > 10_000) {
                        allSmallInt = false;
                        break;
                    }
                    if (iv > maxVal) maxVal = iv;
                }
                if (allSmallInt) {
                    // Upgrade to array counter
                    catInts[t] = new int[n];
                    for (int i = 0; i < n; i++) {
                        catInts[t][i] = (int) features[i][col];
                    }
                    // Store max for counter allocation (use local override)
                    maxCategoryValues[t] = maxVal;
                } else {
                    catLongs[t] = new long[n];
                    for (int i = 0; i < n; i++) {
                        catLongs[t][i] = Double.doubleToRawLongBits(features[i][col]);
                    }
                }
            }
        }

        // ── Array-based max-heap ──
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

        // ── Novelty counters ──
        int[][] arrayCounters = new int[numTowers][];
        @SuppressWarnings("unchecked")
        Map<Long, Integer>[] mapCounters = new HashMap[numTowers];
        for (int t = 0; t < numTowers; t++) {
            if (noveltyColumns[t] < 0) continue;
            if (catInts[t] != null) {
                arrayCounters[t] = new int[maxCategoryValues[t] + 1];
            } else {
                mapCounters[t] = new HashMap<>();
            }
        }

        int[] selected = new int[k];
        boolean[] inSelected = new boolean[n];
        int selectedCount = 0;

        for (int step = 0; step < k; step++) {
            int evals = 0;
            boolean found = false;

            while (heapSize > 0) {
                // Pop top
                int topIdx = heapIdx[0];
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
                    // True best for this step
                    selected[step] = topIdx;
                    found = true;
                    updateCounters(topIdx, catInts, catLongs, arrayCounters, mapCounters);
                    inSelected[topIdx] = true;
                    selectedCount++;
                    break;
                }

                // Recompute diversity for this candidate
                double divScore = computeDiversity(topIdx, catInts, catLongs,
                        arrayCounters, mapCounters, selectedCount);
                double newGain = baseScores[topIdx] + divScore;
                evals++;

                if (evals >= maxEvalsPerPos) {
                    // Budget exhausted: pick best among evaluated this step
                    double bestGain = newGain;
                    int bestIdx = topIdx;

                    for (int h = 0; h < heapSize; h++) {
                        if (heapStep[h] == step && !inSelected[heapIdx[h]] && heapGain[h] > bestGain) {
                            heapGain[heapSize] = bestGain;
                            heapIdx[heapSize] = bestIdx;
                            heapStep[heapSize] = step;
                            heapSize++;
                            siftUp(heapGain, heapIdx, heapStep, heapSize - 1);
                            bestGain = heapGain[h];
                            bestIdx = heapIdx[h];
                            heapSize--;
                            if (h < heapSize) {
                                heapGain[h] = heapGain[heapSize];
                                heapIdx[h] = heapIdx[heapSize];
                                heapStep[h] = heapStep[heapSize];
                                siftDown(heapGain, heapIdx, heapStep, heapSize, h);
                                h--;
                            }
                        }
                    }
                    if (bestIdx != topIdx) {
                        heapGain[heapSize] = newGain;
                        heapIdx[heapSize] = topIdx;
                        heapStep[heapSize] = step;
                        heapSize++;
                        siftUp(heapGain, heapIdx, heapStep, heapSize - 1);
                    }

                    selected[step] = bestIdx;
                    found = true;
                    updateCounters(bestIdx, catInts, catLongs, arrayCounters, mapCounters);
                    inSelected[bestIdx] = true;
                    selectedCount++;
                    break;
                }

                // Push back with updated gain
                heapGain[heapSize] = newGain;
                heapIdx[heapSize] = topIdx;
                heapStep[heapSize] = step;
                heapSize++;
                siftUp(heapGain, heapIdx, heapStep, heapSize - 1);
            }

            if (!found) break;
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
     * Compute diversity score for a candidate using compiled towers + array/map counters.
     */
    private double computeDiversity(int candidateIdx,
                                     int[][] catInts, long[][] catLongs,
                                     int[][] arrayCounters, Map<Long, Integer>[] mapCounters,
                                     int selectedCount) {
        double divScore = 0;
        for (int t = 0; t < numTowers; t++) {
            double gwFeat;
            if (noveltyColumns[t] < 0) {
                gwFeat = 1.0;
            } else if (selectedCount == 0) {
                gwFeat = 1.0;
            } else if (catInts[t] != null) {
                // Array counter: O(1) indexed lookup
                int cat = catInts[t][candidateIdx];
                int matches = arrayCounters[t][cat];
                gwFeat = 1.0 - (double) matches / selectedCount;
            } else {
                // HashMap fallback
                long cat = catLongs[t][candidateIdx];
                int matches = mapCounters[t].getOrDefault(cat, 0);
                gwFeat = 1.0 - (double) matches / selectedCount;
            }
            divScore += compiledTowers[t].eval(gwFeat);
        }
        return divScore;
    }

    /**
     * Update novelty counters when a candidate is selected.
     */
    private void updateCounters(int docIdx,
                                 int[][] catInts, long[][] catLongs,
                                 int[][] arrayCounters, Map<Long, Integer>[] mapCounters) {
        for (int t = 0; t < numTowers; t++) {
            if (noveltyColumns[t] < 0) continue;
            if (catInts[t] != null) {
                arrayCounters[t][catInts[t][docIdx]]++;
            } else {
                long cat = catLongs[t][docIdx];
                mapCounters[t].merge(cat, 1, Integer::sum);
            }
        }
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
