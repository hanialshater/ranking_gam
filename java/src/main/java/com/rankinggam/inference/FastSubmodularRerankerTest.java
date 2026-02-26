package com.rankinggam.inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Tests for {@link FastSubmodularReranker}.
 *
 * <p>Verifies that the optimized reranker produces identical selections to the
 * original {@link SubmodularGamReranker} across various scenarios, and tests
 * edge cases.
 *
 * <p>Run: {@code java com.rankinggam.inference.FastSubmodularRerankerTest}
 */
public final class FastSubmodularRerankerTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        // CompiledConcavePwl tests
        testCompiledConcaveMatchesOriginalK0();
        testCompiledConcaveMatchesOriginalK1();
        testCompiledConcaveMatchesOriginalK4();
        testCompiledConcaveMatchesOriginalK6();
        testCompiledConcaveMatchesOriginalK8Fallback();
        testCompiledConcaveEvaluateAtMax();

        // Reranker tests
        testEmptyInput();
        testSingleDocument();
        testKGreaterThanN();
        testMatchesOriginalUnlimited();
        testMatchesOriginalBudgeted();
        testMatchesOriginalLargeScale();
        testMatchesOriginalManyCategories();
        testPrecomputedBaseScores();
        testSelectionOrderDeterministic();
        testDiversityActuallyDiversifies();
        testArrayCounterMatchesHashMap();

        System.out.println();
        System.out.printf("Results: %d passed, %d failed, %d total%n",
                passed, failed, passed + failed);
        if (failed > 0) System.exit(1);
    }

    // ── Helpers to build test models ──

    private static ConcavePwlFunction[] makeTowers(int count) {
        double[] edges  = {0.0, 0.25, 0.5, 0.75};
        double[] widths = {0.25, 0.25, 0.25, 0.25};
        double[] slopes = {0.8, 0.4, 0.2, 0.1};
        ConcavePwlFunction[] towers = new ConcavePwlFunction[count];
        for (int i = 0; i < count; i++) {
            towers[i] = new ConcavePwlFunction(0.0, edges, widths, slopes, 0.0, 1.0);
        }
        return towers;
    }

    private static DistilledGamModel makeSimpleModel(int numFeatures) {
        List<DistilledGamModel.MainEffect> mains = new ArrayList<>();
        for (int j = 0; j < numFeatures; j++) {
            // Simple linear PWL: y = 0.1 * x
            double[] xk = {-5.0, 5.0};
            double[] yk = {-0.5, 0.5};
            mains.add(new DistilledGamModel.MainEffect(j, new PwlFunction(xk, yk)));
        }
        return new DistilledGamModel(0.0, mains, new ArrayList<>());
    }

    /** Build both rerankers for comparison. catCol and brandCol are the novelty feature columns. */
    private static Object[] buildBoth(DistilledGamModel model, int catCol, int brandCol) {
        ConcavePwlFunction[] towers = makeTowers(2);

        // Original
        DefaultGroupwiseComputer.Spec[] specs = {
            DefaultGroupwiseComputer.Spec.novelty("category_novelty", catCol),
            DefaultGroupwiseComputer.Spec.novelty("brand_novelty", brandCol)
        };
        DefaultGroupwiseComputer computer = new DefaultGroupwiseComputer(specs);
        SubmodularGamReranker original = new SubmodularGamReranker(model, towers, computer);

        // Optimized
        int[] noveltyCols = {catCol, brandCol};
        FastSubmodularReranker fast = new FastSubmodularReranker(model, towers, noveltyCols);

        return new Object[]{original, fast};
    }

    private static double[][] makeFeatures(Random rng, int n, int numFeatures,
                                            int catCol, int numCats, int brandCol, int numBrands) {
        double[][] features = new double[n][numFeatures];
        for (int i = 0; i < n; i++) {
            features[i][catCol] = i % numCats;
            features[i][brandCol] = i % numBrands;
            for (int j = 0; j < numFeatures; j++) {
                if (j != catCol && j != brandCol) {
                    features[i][j] = rng.nextGaussian() * 2.0;
                }
            }
        }
        return features;
    }

    private static void check(String name, boolean condition) {
        if (condition) {
            passed++;
            System.out.println("  PASS: " + name);
        } else {
            failed++;
            System.out.println("  FAIL: " + name);
        }
    }

    // ── CompiledConcavePwl Tests ──

    private static void testCompiledConcaveMatchesOriginalK0() {
        System.out.println("testCompiledConcaveMatchesOriginalK0");
        ConcavePwlFunction src = new ConcavePwlFunction(
                2.5, new double[0], new double[0], new double[0], 0.0, 1.0);
        CompiledConcavePwl compiled = CompiledConcavePwl.compile(src);
        check("K0: constant at 0.0", Math.abs(compiled.evaluate(0.0) - src.evaluate(0.0)) < 1e-12);
        check("K0: constant at 0.5", Math.abs(compiled.evaluate(0.5) - src.evaluate(0.5)) < 1e-12);
        check("K0: constant at 1.0", Math.abs(compiled.evaluate(1.0) - src.evaluate(1.0)) < 1e-12);
    }

    private static void testCompiledConcaveMatchesOriginalK1() {
        System.out.println("testCompiledConcaveMatchesOriginalK1");
        double[] edges = {0.2};
        double[] widths = {0.6};
        double[] slopes = {1.5};
        ConcavePwlFunction src = new ConcavePwlFunction(0.1, edges, widths, slopes, 0.0, 1.0);
        CompiledConcavePwl compiled = CompiledConcavePwl.compile(src);
        double maxErr = checkManyPoints(src, compiled, 0.0, 1.0, 1000);
        check("K1: max error = " + maxErr, maxErr < 1e-12);
    }

    private static void testCompiledConcaveMatchesOriginalK4() {
        System.out.println("testCompiledConcaveMatchesOriginalK4");
        double[] edges  = {0.0, 0.25, 0.5, 0.75};
        double[] widths = {0.25, 0.25, 0.25, 0.25};
        double[] slopes = {0.8, 0.4, 0.2, 0.1};
        ConcavePwlFunction src = new ConcavePwlFunction(0.0, edges, widths, slopes, 0.0, 1.0);
        CompiledConcavePwl compiled = CompiledConcavePwl.compile(src);
        double maxErr = checkManyPoints(src, compiled, -0.5, 1.5, 10000);
        check("K4: max error = " + maxErr, maxErr < 1e-12);
    }

    private static void testCompiledConcaveMatchesOriginalK6() {
        System.out.println("testCompiledConcaveMatchesOriginalK6");
        double[] edges  = {0.0, 0.1, 0.3, 0.5, 0.7, 0.9};
        double[] widths = {0.1, 0.2, 0.2, 0.2, 0.2, 0.1};
        double[] slopes = {1.0, 0.8, 0.6, 0.4, 0.2, 0.1};
        ConcavePwlFunction src = new ConcavePwlFunction(0.5, edges, widths, slopes, 0.0, 1.0);
        CompiledConcavePwl compiled = CompiledConcavePwl.compile(src);
        double maxErr = checkManyPoints(src, compiled, 0.0, 1.0, 10000);
        check("K6: max error = " + maxErr, maxErr < 1e-12);
    }

    private static void testCompiledConcaveMatchesOriginalK8Fallback() {
        System.out.println("testCompiledConcaveMatchesOriginalK8Fallback");
        double[] edges  = new double[8];
        double[] widths = new double[8];
        double[] slopes = new double[8];
        for (int i = 0; i < 8; i++) {
            edges[i] = i * 0.125;
            widths[i] = 0.125;
            slopes[i] = 1.0 - i * 0.1;
        }
        ConcavePwlFunction src = new ConcavePwlFunction(0.0, edges, widths, slopes, 0.0, 1.0);
        CompiledConcavePwl compiled = CompiledConcavePwl.compile(src);
        double maxErr = checkManyPoints(src, compiled, 0.0, 1.0, 10000);
        check("K8 fallback: max error = " + maxErr, maxErr < 1e-12);
    }

    private static void testCompiledConcaveEvaluateAtMax() {
        System.out.println("testCompiledConcaveEvaluateAtMax");
        double[] edges  = {0.0, 0.25, 0.5, 0.75};
        double[] widths = {0.25, 0.25, 0.25, 0.25};
        double[] slopes = {0.8, 0.4, 0.2, 0.1};
        ConcavePwlFunction src = new ConcavePwlFunction(0.0, edges, widths, slopes, 0.0, 1.0);
        CompiledConcavePwl compiled = CompiledConcavePwl.compile(src);
        check("evaluateAtMax matches",
                Math.abs(compiled.evaluateAtMax() - src.evaluateAtMax()) < 1e-12);
        check("evaluateAtMax matches evaluate(1.0)",
                Math.abs(compiled.evaluateAtMax() - compiled.evaluate(1.0)) < 1e-12);
    }

    private static double checkManyPoints(ConcavePwlFunction src, CompiledConcavePwl compiled,
                                           double lo, double hi, int n) {
        double maxErr = 0;
        for (int i = 0; i <= n; i++) {
            double x = lo + (hi - lo) * i / n;
            double expected = src.evaluate(x);
            double actual = compiled.evaluate(x);
            maxErr = Math.max(maxErr, Math.abs(expected - actual));
        }
        return maxErr;
    }

    // ── Reranker Tests ──

    private static void testEmptyInput() {
        System.out.println("testEmptyInput");
        DistilledGamModel model = makeSimpleModel(3);
        int[] novCols = {0, 1};
        FastSubmodularReranker fast = new FastSubmodularReranker(model, makeTowers(2), novCols);

        int[] result = fast.rerank(new double[0][], 10);
        check("empty features returns empty", result.length == 0);

        result = fast.rerank(new double[5][3], 0);
        check("k=0 returns empty", result.length == 0);
    }

    private static void testSingleDocument() {
        System.out.println("testSingleDocument");
        DistilledGamModel model = makeSimpleModel(3);
        Object[] both = buildBoth(model, 0, 1);
        SubmodularGamReranker original = (SubmodularGamReranker) both[0];
        FastSubmodularReranker fast = (FastSubmodularReranker) both[1];

        double[][] features = {{1.0, 2.0, 3.0}};
        int[] origResult = original.rerank(features, 1);
        int[] fastResult = fast.rerank(features, 1);

        check("single doc: both return [0]",
                Arrays.equals(origResult, fastResult) && origResult[0] == 0);
    }

    private static void testKGreaterThanN() {
        System.out.println("testKGreaterThanN");
        DistilledGamModel model = makeSimpleModel(4);
        Object[] both = buildBoth(model, 0, 1);
        SubmodularGamReranker original = (SubmodularGamReranker) both[0];
        FastSubmodularReranker fast = (FastSubmodularReranker) both[1];

        Random rng = new Random(42);
        double[][] features = makeFeatures(rng, 5, 4, 0, 3, 1, 2);
        int[] origResult = original.rerank(features, 100);
        int[] fastResult = fast.rerank(features, 100);

        check("k>n: same length", origResult.length == fastResult.length);
        check("k>n: same selection", Arrays.equals(origResult, fastResult));
    }

    private static void testMatchesOriginalUnlimited() {
        System.out.println("testMatchesOriginalUnlimited (n=100, k=20)");
        DistilledGamModel model = makeSimpleModel(10);
        Object[] both = buildBoth(model, 0, 1);
        SubmodularGamReranker original = (SubmodularGamReranker) both[0];
        FastSubmodularReranker fast = (FastSubmodularReranker) both[1];

        Random rng = new Random(123);
        double[][] features = makeFeatures(rng, 100, 10, 0, 5, 1, 3);

        int[] origResult = original.rerank(features, 20);
        int[] fastResult = fast.rerank(features, 20);

        check("unlimited: same length", origResult.length == fastResult.length);
        check("unlimited: identical selections", Arrays.equals(origResult, fastResult));
    }

    private static void testMatchesOriginalBudgeted() {
        System.out.println("testMatchesOriginalBudgeted (n=200, k=20, budget=5)");
        DistilledGamModel model = makeSimpleModel(10);
        Object[] both = buildBoth(model, 0, 1);
        SubmodularGamReranker original = (SubmodularGamReranker) both[0];
        FastSubmodularReranker fast = (FastSubmodularReranker) both[1];

        Random rng = new Random(456);
        double[][] features = makeFeatures(rng, 200, 10, 0, 7, 1, 4);

        int[] origResult = original.rerank(features, 20, 5);
        int[] fastResult = fast.rerank(features, 20, 5);

        check("budgeted: same length", origResult.length == fastResult.length);
        check("budgeted: identical selections", Arrays.equals(origResult, fastResult));
    }

    private static void testMatchesOriginalLargeScale() {
        System.out.println("testMatchesOriginalLargeScale (n=1000, k=40)");
        DistilledGamModel model = makeSimpleModel(20);
        Object[] both = buildBoth(model, 0, 1);
        SubmodularGamReranker original = (SubmodularGamReranker) both[0];
        FastSubmodularReranker fast = (FastSubmodularReranker) both[1];

        Random rng = new Random(789);
        double[][] features = makeFeatures(rng, 1000, 20, 0, 10, 1, 5);

        int[] origResult = original.rerank(features, 40);
        int[] fastResult = fast.rerank(features, 40);

        check("large scale: same length", origResult.length == fastResult.length);
        check("large scale: identical selections", Arrays.equals(origResult, fastResult));
    }

    private static void testMatchesOriginalManyCategories() {
        System.out.println("testMatchesOriginalManyCategories (n=500, k=30, 50 cats)");
        DistilledGamModel model = makeSimpleModel(10);
        Object[] both = buildBoth(model, 0, 1);
        SubmodularGamReranker original = (SubmodularGamReranker) both[0];
        FastSubmodularReranker fast = (FastSubmodularReranker) both[1];

        Random rng = new Random(999);
        double[][] features = makeFeatures(rng, 500, 10, 0, 50, 1, 20);

        int[] origResult = original.rerank(features, 30);
        int[] fastResult = fast.rerank(features, 30);

        check("many categories: same length", origResult.length == fastResult.length);
        check("many categories: identical selections", Arrays.equals(origResult, fastResult));
    }

    private static void testPrecomputedBaseScores() {
        System.out.println("testPrecomputedBaseScores");
        DistilledGamModel model = makeSimpleModel(10);
        int[] novCols = {0, 1};
        FastSubmodularReranker fast = new FastSubmodularReranker(model, makeTowers(2), novCols);

        Random rng = new Random(42);
        double[][] features = makeFeatures(rng, 100, 10, 0, 5, 1, 3);

        // Compute base scores externally
        double[] baseScores = model.score(features);

        int[] autoResult = fast.rerank(features, 20);
        int[] precompResult = fast.rerank(features, baseScores, 20, 0);

        check("precomputed scores: identical to auto-computed",
                Arrays.equals(autoResult, precompResult));
    }

    private static void testSelectionOrderDeterministic() {
        System.out.println("testSelectionOrderDeterministic");
        DistilledGamModel model = makeSimpleModel(10);
        int[] novCols = {0, 1};
        FastSubmodularReranker fast = new FastSubmodularReranker(model, makeTowers(2), novCols);

        Random rng = new Random(42);
        double[][] features = makeFeatures(rng, 200, 10, 0, 5, 1, 3);

        int[] run1 = fast.rerank(features, 20);
        int[] run2 = fast.rerank(features, 20);

        check("deterministic: two runs produce identical results",
                Arrays.equals(run1, run2));
    }

    private static void testDiversityActuallyDiversifies() {
        System.out.println("testDiversityActuallyDiversifies");
        // 10 docs: 5 in cat 0 (high base score), 5 in cat 1 (lower base score)
        // Without diversity, greedy would pick all cat-0 first.
        // With diversity, it should interleave.
        int nf = 3;
        double[][] features = new double[10][nf];
        for (int i = 0; i < 5; i++) {
            features[i][0] = 0; // cat 0
            features[i][1] = 0; // brand 0
            features[i][2] = 10.0 - i * 0.1; // high base score
        }
        for (int i = 5; i < 10; i++) {
            features[i][0] = 1; // cat 1
            features[i][1] = 1; // brand 1
            features[i][2] = 8.0 - (i - 5) * 0.1; // lower base score
        }

        // Model with only feature 2 contributing to base score
        List<DistilledGamModel.MainEffect> mains = new ArrayList<>();
        mains.add(new DistilledGamModel.MainEffect(2, new PwlFunction(
                new double[]{-10, 20}, new double[]{-10, 20})));
        DistilledGamModel model = new DistilledGamModel(0, mains, new ArrayList<>());

        int[] novCols = {0, 1};
        FastSubmodularReranker fast = new FastSubmodularReranker(model, makeTowers(2), novCols);

        int[] selected = fast.rerank(features, 6);

        // Count how many from each category in the top 6
        int cat0 = 0, cat1 = 0;
        for (int idx : selected) {
            if (features[idx][0] == 0) cat0++;
            else cat1++;
        }

        check("diversity: not all from same category (cat0=" + cat0 + ", cat1=" + cat1 + ")",
                cat0 > 0 && cat1 > 0);
        check("diversity: cat-1 appears in top 6 despite lower base score",
                countCat(features, selected, 1, 6) > 0);
    }

    private static void testArrayCounterMatchesHashMap() {
        System.out.println("testArrayCounterMatchesHashMap (n=500, k=30)");
        DistilledGamModel model = makeSimpleModel(10);
        ConcavePwlFunction[] towers = makeTowers(2);
        int catCol = 0, brandCol = 1;
        int[] novCols = {catCol, brandCol};

        // HashMap version (default, maxCats = -1)
        FastSubmodularReranker hashMapReranker = new FastSubmodularReranker(model, towers, novCols);

        // Array counter version (explicit maxCategoryValues)
        int[] maxCats = {50, 20};
        FastSubmodularReranker arrayReranker = new FastSubmodularReranker(model, towers, novCols, maxCats);

        Random rng = new Random(777);
        double[][] features = makeFeatures(rng, 500, 10, catCol, 50, brandCol, 20);

        int[] hashResult = hashMapReranker.rerank(features, 30);
        int[] arrayResult = arrayReranker.rerank(features, 30);

        check("array vs hashmap: same length", hashResult.length == arrayResult.length);
        check("array vs hashmap: identical selections", Arrays.equals(hashResult, arrayResult));
    }

    private static int countCat(double[][] features, int[] selected, double catVal, int topK) {
        int count = 0;
        for (int i = 0; i < Math.min(topK, selected.length); i++) {
            if (features[selected[i]][0] == catVal) count++;
        }
        return count;
    }
}
