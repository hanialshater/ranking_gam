package com.rankinggam.inference;

import java.util.PriorityQueue;

/**
 * Submodular GAM reranker with Minoux lazy greedy acceleration.
 *
 * <p>Scoring function:
 * <pre>
 *   score(item_i | S) = base_score(i) + sum_k diversity_tower_k(gw_feature_k(i, S))
 * </pre>
 *
 * <p>The base scores are computed once (via {@link DistilledGamModel}) and cached.
 * Diversity contributions use concave PWL towers ({@link ConcavePwlFunction})
 * applied to groupwise features computed by {@link GroupwiseFeatureComputer}.
 *
 * <p><b>Minoux lazy greedy (1978):</b> Uses a priority queue of upper bounds.
 * Due to submodularity, a candidate's marginal gain can only decrease as the
 * selected set grows. So previous gains are valid upper bounds. We only
 * recompute when a candidate reaches the top of the queue.
 *
 * <p><b>Max eval budget:</b> At each position, limit the number of candidate
 * re-evaluations. If the budget is exhausted, pick the best among those
 * evaluated. This caps worst-case cost to O(k * maxEvalsPerPos).
 *
 * <p>Typical performance: for 10k candidates, k=40, maxEvalsPerPos=10,
 * the reranker needs ~40*10 = 400 diversity evaluations (vs 40*10000 = 400k naive).
 */
public final class SubmodularGamReranker {

    private final DistilledGamModel baseModel;
    private final ConcavePwlFunction[] diversityTowers;
    private final GroupwiseFeatureComputer featureComputer;

    // Upper-bound on diversity score (sum of all towers at x_max)
    private final double maxDiversityScore;

    public SubmodularGamReranker(DistilledGamModel baseModel,
                                  ConcavePwlFunction[] diversityTowers,
                                  GroupwiseFeatureComputer featureComputer) {
        this.baseModel = baseModel;
        this.diversityTowers = diversityTowers;
        this.featureComputer = featureComputer;

        double maxDiv = 0;
        for (ConcavePwlFunction tower : diversityTowers) {
            maxDiv += tower.evaluateAtMax();
        }
        this.maxDiversityScore = maxDiv;
    }

    /**
     * Rerank documents using lazy greedy submodular maximization.
     *
     * @param features       [numDocs][numFeatures] feature matrix
     * @param k              number of items to select
     * @param maxEvalsPerPos max candidate re-evaluations per position (0 = unlimited)
     * @return selected indices in selection order, length min(k, numDocs)
     */
    public int[] rerank(double[][] features, int k, int maxEvalsPerPos) {
        final int n = features.length;
        if (k <= 0 || n == 0) return new int[0];
        k = Math.min(k, n);
        if (maxEvalsPerPos <= 0) maxEvalsPerPos = n;

        // Compute base scores once
        double[] baseScores = baseModel.score(features);

        // Selected set tracking
        int[] selected = new int[k];
        boolean[] inSelected = new boolean[n];
        int[] selectedSoFar = new int[0]; // grows each step

        // Priority queue: max-heap by gain
        PriorityQueue<Candidate> pq = new PriorityQueue<>(n);
        for (int i = 0; i < n; i++) {
            // Initial upper bound: base + max possible diversity
            pq.add(new Candidate(i, baseScores[i] + maxDiversityScore, -1));
        }

        for (int step = 0; step < k; step++) {
            int evals = 0;
            Candidate best = null;

            while (!pq.isEmpty()) {
                Candidate top = pq.poll();

                // Skip if already selected (shouldn't happen, but safety)
                if (inSelected[top.index]) continue;

                // If computed for this step, it's the true best
                if (top.computedAtStep == step) {
                    best = top;
                    break;
                }

                // Recompute marginal gain for current selected set
                double[] gwFeats = featureComputer.compute(
                        top.index, selectedSoFar, features);
                double divScore = 0;
                for (int t = 0; t < diversityTowers.length; t++) {
                    divScore += diversityTowers[t].evaluate(gwFeats[t]);
                }
                top.gain = baseScores[top.index] + divScore;
                top.computedAtStep = step;
                evals++;

                // Check budget: if exhausted, pick best among evaluated
                if (evals >= maxEvalsPerPos) {
                    // top was just evaluated; find best among it and any
                    // previously evaluated candidates still in the queue
                    best = top;
                    // Drain candidates evaluated this step that might be better
                    while (!pq.isEmpty()) {
                        Candidate peek = pq.peek();
                        if (peek.computedAtStep == step) {
                            Candidate c = pq.poll();
                            if (!inSelected[c.index] && c.gain > best.gain) {
                                pq.add(best); // put previous best back
                                best = c;
                            } else {
                                pq.add(c); // put it back
                                break; // rest are lower bound, stop
                            }
                        } else {
                            break; // not evaluated this step, keep as is
                        }
                    }
                    // Push the just-evaluated candidate back if it's not best
                    if (best != top) {
                        pq.add(top);
                    }
                    break;
                }

                // Push back with updated gain
                pq.add(top);
            }

            if (best == null) break; // shouldn't happen if n > step

            selected[step] = best.index;
            inSelected[best.index] = true;

            // Update selectedSoFar array
            int[] newSel = new int[step + 1];
            System.arraycopy(selectedSoFar, 0, newSel, 0, step);
            newSel[step] = best.index;
            selectedSoFar = newSel;
        }

        return selected;
    }

    /**
     * Rerank with unlimited eval budget (pure Minoux lazy greedy).
     */
    public int[] rerank(double[][] features, int k) {
        return rerank(features, k, 0);
    }

    /**
     * Rerank and return both indices and scores at selection.
     */
    public RerankResult rerankWithScores(double[][] features, int k, int maxEvalsPerPos) {
        final int n = features.length;
        if (k <= 0 || n == 0) return new RerankResult(new int[0], new double[0]);
        k = Math.min(k, n);
        if (maxEvalsPerPos <= 0) maxEvalsPerPos = n;

        double[] baseScores = baseModel.score(features);

        int[] selected = new int[k];
        double[] scores = new double[k];
        boolean[] inSelected = new boolean[n];
        int[] selectedSoFar = new int[0];

        PriorityQueue<Candidate> pq = new PriorityQueue<>(n);
        for (int i = 0; i < n; i++) {
            pq.add(new Candidate(i, baseScores[i] + maxDiversityScore, -1));
        }

        for (int step = 0; step < k; step++) {
            int evals = 0;
            Candidate best = null;

            while (!pq.isEmpty()) {
                Candidate top = pq.poll();
                if (inSelected[top.index]) continue;

                if (top.computedAtStep == step) {
                    best = top;
                    break;
                }

                double[] gwFeats = featureComputer.compute(
                        top.index, selectedSoFar, features);
                double divScore = 0;
                for (int t = 0; t < diversityTowers.length; t++) {
                    divScore += diversityTowers[t].evaluate(gwFeats[t]);
                }
                top.gain = baseScores[top.index] + divScore;
                top.computedAtStep = step;
                evals++;

                if (evals >= maxEvalsPerPos) {
                    best = top;
                    while (!pq.isEmpty()) {
                        Candidate peek = pq.peek();
                        if (peek.computedAtStep == step) {
                            Candidate c = pq.poll();
                            if (!inSelected[c.index] && c.gain > best.gain) {
                                pq.add(best);
                                best = c;
                            } else {
                                pq.add(c);
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if (best != top) {
                        pq.add(top);
                    }
                    break;
                }

                pq.add(top);
            }

            if (best == null) break;

            selected[step] = best.index;
            scores[step] = best.gain;
            inSelected[best.index] = true;

            int[] newSel = new int[step + 1];
            System.arraycopy(selectedSoFar, 0, newSel, 0, step);
            newSel[step] = best.index;
            selectedSoFar = newSel;
        }

        return new RerankResult(selected, scores);
    }

    /** Result of reranking with scores. */
    public static final class RerankResult {
        public final int[] indices;
        public final double[] scores;

        RerankResult(int[] indices, double[] scores) {
            this.indices = indices;
            this.scores = scores;
        }
    }

    /** Priority queue entry. Mutable gain for lazy updates. */
    private static final class Candidate implements Comparable<Candidate> {
        final int index;
        double gain;
        int computedAtStep;

        Candidate(int index, double gain, int computedAtStep) {
            this.index = index;
            this.gain = gain;
            this.computedAtStep = computedAtStep;
        }

        @Override
        public int compareTo(Candidate o) {
            return Double.compare(o.gain, this.gain); // max-heap
        }
    }
}
