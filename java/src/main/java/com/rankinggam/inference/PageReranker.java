package com.rankinggam.inference;

import java.util.Arrays;

/**
 * Hierarchical parallel greedy reranker for paginated results.
 *
 * <p>Real-world scenario: 1000+ items pre-sorted by relevance, split into pages
 * of ~100. The reranker diversifies within each page and smooths page boundaries.
 *
 * <h3>Two-pass approach:</h3>
 * <ol>
 *   <li><b>Pass 1 — Per-page greedy:</b> Each page is reranked
 *       independently using {@link FastSubmodularReranker}'s lazy greedy heap.
 *       Items stay within their page — a page-0 item can't migrate to page 5.
 *   <li><b>Pass 2 — Boundary refinement (sequential, budget-limited):</b>
 *       For each boundary between adjacent pages, takes the last {@code budget}
 *       items of page p and the first {@code budget} of page p+1, and re-runs
 *       greedy on this 2*budget window. Diversity state is built from all items
 *       before the window. This smooths transitions and makes the result
 *       "almost sorted" with local diversity boosts near boundaries.
 * </ol>
 *
 * <h3>Methods:</h3>
 * <ul>
 *   <li>{@link #rerank} — single page (delegates to FastSubmodularReranker)
 *   <li>{@link #rerankAllPages} — pass 1 only: per-page greedy
 *   <li>{@link #rerankWithBoundaryPass} — pass 1 + pass 2
 * </ul>
 */
public final class PageReranker {

    private final DistilledGamModel baseModel;
    private final ConcavePwlFunction[] diversityTowers;
    private final int[] noveltyColumns;
    private final int[] maxCategoryValues;

    // Shared reranker for single-threaded single-page use
    private final FastSubmodularReranker sharedReranker;

    // Compiled towers for boundary-pass flat scan
    private final CompiledConcavePwl.ConcaveEval[] compiledTowers;
    private final int numTowers;
    private final double maxDiversityScore;

    /**
     * @param baseModel         base GAM scoring model
     * @param diversityTowers   concave PWL diversity towers
     * @param noveltyColumns    per-tower: feature column index for novelty
     * @param maxCategoryValues per-tower: max integer category value for array counters
     */
    public PageReranker(DistilledGamModel baseModel,
                        ConcavePwlFunction[] diversityTowers,
                        int[] noveltyColumns,
                        int[] maxCategoryValues) {
        this.baseModel = baseModel;
        this.diversityTowers = diversityTowers.clone();
        this.numTowers = diversityTowers.length;
        this.noveltyColumns = noveltyColumns.clone();
        this.maxCategoryValues = maxCategoryValues.clone();

        this.sharedReranker = new FastSubmodularReranker(
                baseModel, diversityTowers, noveltyColumns, maxCategoryValues);

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
     * Constructor without explicit maxCategoryValues (auto-detected).
     */
    public PageReranker(DistilledGamModel baseModel,
                        ConcavePwlFunction[] diversityTowers,
                        int[] noveltyColumns) {
        this(baseModel, diversityTowers, noveltyColumns, defaultMaxCats(diversityTowers.length));
    }

    private static int[] defaultMaxCats(int n) {
        int[] m = new int[n];
        Arrays.fill(m, -1);
        return m;
    }

    // ── Single-page rerank (delegates to FastSubmodularReranker) ──

    /**
     * Rerank a single page. Uses the heap-based lazy greedy internally.
     *
     * @param features   [n][numFeatures] feature matrix
     * @param baseScores [n] precomputed base scores
     * @param k          items to select
     * @return selected indices in selection order
     */
    public int[] rerank(double[][] features, double[] baseScores, int k) {
        return sharedReranker.rerank(features, baseScores, k, 0);
    }

    /**
     * Rerank computing base scores internally.
     */
    public int[] rerank(double[][] features, int k) {
        return rerank(features, baseModel.score(features), k);
    }

    // ── Pass 1: Per-page greedy ──

    /**
     * Split items into pages and run greedy within each page.
     *
     * <p>Items are assumed to be pre-sorted by relevance (page 0 = most relevant).
     * Each page is reranked independently — items stay within their page.
     *
     * @param features   [totalDocs][numFeatures] — pre-sorted by relevance
     * @param baseScores [totalDocs]
     * @param pageSize   items per page (e.g., 100)
     * @param k          items to select per page
     * @return flattened ordering with global indices
     */
    public int[] rerankAllPages(double[][] features, double[] baseScores,
                                int pageSize, int k) {
        final int totalDocs = features.length;
        final int numPages = (totalDocs + pageSize - 1) / pageSize;
        int[][] pageResults = new int[numPages][];

        for (int p = 0; p < numPages; p++) {
            int start = p * pageSize;
            int end = Math.min(start + pageSize, totalDocs);
            int pageLen = end - start;
            int pageK = Math.min(k, pageLen);

            // Slice features and scores for this page
            double[][] pageFeat = new double[pageLen][];
            double[] pageScores = new double[pageLen];
            for (int i = 0; i < pageLen; i++) {
                pageFeat[i] = features[start + i];
                pageScores[i] = baseScores[start + i];
            }

            // Fresh instance per page: auto-detected maxCategoryValues vary per page
            FastSubmodularReranker pageRanker = new FastSubmodularReranker(
                    baseModel, diversityTowers, noveltyColumns, maxCategoryValues);
            int[] localOrder = pageRanker.rerank(pageFeat, pageScores, pageK, 0);

            // Convert local indices to global
            int[] globalOrder = new int[localOrder.length];
            for (int i = 0; i < localOrder.length; i++) {
                globalOrder[i] = localOrder[i] + start;
            }
            pageResults[p] = globalOrder;
        }

        return flatten(pageResults);
    }

    /**
     * Rerank all pages, computing base scores internally.
     */
    public int[] rerankAllPages(double[][] features, int pageSize, int k) {
        return rerankAllPages(features, baseModel.score(features), pageSize, k);
    }

    // ── Pass 2: Cross-page boundary refinement ──

    /**
     * Two-pass reranking: parallel per-page greedy + cross-page boundary refinement.
     *
     * <p><b>Pass 1:</b> Parallel greedy within each page.
     *
     * <p><b>Pass 2:</b> For each boundary between adjacent pages, take the last
     * {@code budget} items of page p and the first {@code budget} items of page p+1,
     * and re-run greedy on this 2*budget window. Diversity state is accumulated
     * from items before the window. Flat scan is used for these small windows
     * (typically 2*budget ≈ 20 items).
     *
     * @param features   [totalDocs][numFeatures] — pre-sorted by relevance
     * @param baseScores [totalDocs]
     * @param pageSize   items per page (e.g., 100)
     * @param k          items to select per page
     * @param budget     boundary window half-size (e.g., 10 → 20-item windows)
     * @return flattened ordering with boundary refinement applied
     */
    public int[] rerankWithBoundaryPass(double[][] features, double[] baseScores,
                                        int pageSize, int k, int budget) {
        // Pass 1
        int[] fullOrder = rerankAllPages(features, baseScores, pageSize, k);

        if (budget <= 0) return fullOrder;

        final int totalSelected = fullOrder.length;
        final int totalDocs = features.length;
        final int numPages = (totalDocs + pageSize - 1) / pageSize;

        // Pass 2: boundary refinement
        // Build counters from scratch and walk through the ordering.
        // For each boundary, snapshot the counter state, re-greedy the window,
        // and continue accumulating.

        // Find global max category per tower (for counter allocation)
        int[] globalMaxCat = new int[numTowers];
        for (int idx = 0; idx < totalSelected; idx++) {
            int doc = fullOrder[idx];
            for (int t = 0; t < numTowers; t++) {
                int cat = (int) features[doc][noveltyColumns[t]];
                if (cat > globalMaxCat[t]) globalMaxCat[t] = cat;
            }
        }

        // Running counters — accumulate as we walk through the ordering
        int[][] counters = new int[numTowers][];
        for (int t = 0; t < numTowers; t++) {
            counters[t] = new int[globalMaxCat[t] + 1];
        }

        int accumulated = 0; // how many items we've accumulated into counters

        for (int p = 0; p < numPages - 1; p++) {
            int pageEnd = (p + 1) * k; // boundary position in fullOrder
            if (pageEnd >= totalSelected) break;

            int windowStart = Math.max(0, pageEnd - budget);
            int windowEnd = Math.min(totalSelected, pageEnd + budget);
            int windowSize = windowEnd - windowStart;
            if (windowSize <= 1) {
                // Accumulate up to windowEnd for next boundary
                for (int idx = accumulated; idx < windowEnd; idx++) {
                    int doc = fullOrder[idx];
                    for (int t = 0; t < numTowers; t++) {
                        counters[t][(int) features[doc][noveltyColumns[t]]]++;
                    }
                }
                accumulated = windowEnd;
                continue;
            }

            // Accumulate counters up to windowStart
            for (int idx = accumulated; idx < windowStart; idx++) {
                int doc = fullOrder[idx];
                for (int t = 0; t < numTowers; t++) {
                    counters[t][(int) features[doc][noveltyColumns[t]]]++;
                }
            }
            accumulated = windowStart;

            // Snapshot counters for the window re-greedy
            int[][] windowCounters = new int[numTowers][];
            for (int t = 0; t < numTowers; t++) {
                windowCounters[t] = counters[t].clone();
            }

            // Collect window items
            int[] windowItems = new int[windowSize];
            System.arraycopy(fullOrder, windowStart, windowItems, 0, windowSize);

            // Re-greedy over window (flat scan — small window, no heap needed)
            int[] reordered = rerankWindow(windowItems, features, baseScores,
                                           windowCounters, windowStart);

            // Write reordered items back
            System.arraycopy(reordered, 0, fullOrder, windowStart, windowSize);

            // Accumulate the (reordered) window items into running counters
            for (int idx = windowStart; idx < windowEnd; idx++) {
                int doc = fullOrder[idx];
                for (int t = 0; t < numTowers; t++) {
                    counters[t][(int) features[doc][noveltyColumns[t]]]++;
                }
            }
            accumulated = windowEnd;
        }

        return fullOrder;
    }

    /**
     * Two-pass reranking, computing base scores internally.
     */
    public int[] rerankWithBoundaryPass(double[][] features, int pageSize,
                                        int k, int budget) {
        return rerankWithBoundaryPass(features, baseModel.score(features),
                                      pageSize, k, budget);
    }

    // ── Internal: flat-scan greedy for small boundary windows ──

    /**
     * Greedy rerank a small window of items, given pre-built diversity state.
     * Uses flat scan (no heap) — optimal for windows of ~20 items.
     */
    private int[] rerankWindow(int[] items, double[][] features, double[] baseScores,
                               int[][] counters, int selectedBefore) {
        int windowSize = items.length;
        boolean[] used = new boolean[windowSize];
        int[] result = new int[windowSize];
        int selectedCount = selectedBefore;

        // Pre-extract category values for window items
        int[][] catInts = new int[numTowers][windowSize];
        for (int t = 0; t < numTowers; t++) {
            int col = noveltyColumns[t];
            for (int w = 0; w < windowSize; w++) {
                catInts[t][w] = (int) features[items[w]][col];
            }
        }

        for (int step = 0; step < windowSize; step++) {
            double bestGain = Double.NEGATIVE_INFINITY;
            int bestW = 0;

            for (int w = 0; w < windowSize; w++) {
                if (used[w]) continue;

                double div = 0;
                if (selectedCount > 0) {
                    for (int t = 0; t < numTowers; t++) {
                        int cat = catInts[t][w];
                        int matches = counters[t][cat];
                        double novelty = 1.0 - (double) matches / selectedCount;
                        div += compiledTowers[t].eval(novelty);
                    }
                } else {
                    div = maxDiversityScore;
                }

                double gain = baseScores[items[w]] + div;
                if (gain > bestGain) {
                    bestGain = gain;
                    bestW = w;
                }
            }

            result[step] = items[bestW];
            used[bestW] = true;
            for (int t = 0; t < numTowers; t++) {
                counters[t][catInts[t][bestW]]++;
            }
            selectedCount++;
        }

        return result;
    }

    private static int[] flatten(int[][] pages) {
        int total = 0;
        for (int[] page : pages) total += page.length;
        int[] result = new int[total];
        int pos = 0;
        for (int[] page : pages) {
            System.arraycopy(page, 0, result, pos, page.length);
            pos += page.length;
        }
        return result;
    }
}
