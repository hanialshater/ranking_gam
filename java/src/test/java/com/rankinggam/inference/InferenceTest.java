package com.rankinggam.inference;

import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

/**
 * Self-contained test runner for distilled GAM inference.
 * Does not require JUnit — run directly with {@code java InferenceTest}.
 */
public class InferenceTest {

    private static int passed = 0;
    private static int failed = 0;

    private static void assertEquals(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            System.err.println("FAIL: " + msg + " — expected " + expected + " but got " + actual);
            failed++;
        } else {
            passed++;
        }
    }

    private static void assertTrue(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            failed++;
        } else {
            passed++;
        }
    }

    // ── PwlFunction tests ──

    static void testPwlLinear() {
        PwlFunction f = new PwlFunction(new double[]{0, 1}, new double[]{0, 2});
        assertEquals(0.0, f.evaluate(0.0), 1e-9, "pwl linear at 0");
        assertEquals(1.0, f.evaluate(0.5), 1e-9, "pwl linear at 0.5");
        assertEquals(2.0, f.evaluate(1.0), 1e-9, "pwl linear at 1");
    }

    static void testPwlClamp() {
        PwlFunction f = new PwlFunction(new double[]{1, 2}, new double[]{10, 20});
        assertEquals(10.0, f.evaluate(0.0), 1e-9, "pwl clamp below");
        assertEquals(20.0, f.evaluate(3.0), 1e-9, "pwl clamp above");
    }

    static void testPwlMultiSegment() {
        PwlFunction f = new PwlFunction(
                new double[]{0, 1, 2, 3}, new double[]{0, 1, 1, 2});
        assertEquals(0.5, f.evaluate(0.5), 1e-9, "pwl seg1 midpoint");
        assertEquals(1.0, f.evaluate(1.5), 1e-9, "pwl seg2 flat");
        assertEquals(1.5, f.evaluate(2.5), 1e-9, "pwl seg3 midpoint");
    }

    static void testPwlBatch() {
        PwlFunction f = new PwlFunction(new double[]{0, 1}, new double[]{0, 2});
        double[] result = f.evaluate(new double[]{0, 0.25, 0.5, 0.75, 1.0});
        double[] expected = {0, 0.5, 1.0, 1.5, 2.0};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-9, "pwl batch[" + i + "]");
        }
    }

    // ── BilinearGridFunction tests ──

    static void testBilinearCorners() {
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0, 1}, new double[]{0, 1},
                new double[][]{{0, 1}, {2, 3}});
        assertEquals(0.0, f.evaluate(0, 0), 1e-9, "grid corner (0,0)");
        assertEquals(1.0, f.evaluate(0, 1), 1e-9, "grid corner (0,1)");
        assertEquals(2.0, f.evaluate(1, 0), 1e-9, "grid corner (1,0)");
        assertEquals(3.0, f.evaluate(1, 1), 1e-9, "grid corner (1,1)");
    }

    static void testBilinearCenter() {
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0, 1}, new double[]{0, 1},
                new double[][]{{0, 1}, {2, 3}});
        assertEquals(1.5, f.evaluate(0.5, 0.5), 1e-9, "grid center");
    }

    static void testBilinearEdge() {
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0, 1}, new double[]{0, 1},
                new double[][]{{0, 1}, {2, 3}});
        assertEquals(0.5, f.evaluate(0, 0.5), 1e-9, "grid edge x1=0");
        assertEquals(1.0, f.evaluate(0.5, 0), 1e-9, "grid edge x2=0");
    }

    static void testBilinearClamp() {
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0, 1}, new double[]{0, 1},
                new double[][]{{0, 1}, {2, 3}});
        assertEquals(0.0, f.evaluate(-1, -1), 1e-9, "grid clamp below");
        assertEquals(3.0, f.evaluate(2, 2), 1e-9, "grid clamp above");
    }

    // ── DistilledGamModel tests ──

    static void testModelFromJson() throws Exception {
        InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        assertTrue(is != null, "test_model.json found on classpath");
        DistilledGamModel model = DistilledGamLoader.fromJson(is);
        is.close();

        assertEquals(0.5, model.bias(), 1e-9, "model bias");
        assertTrue(model.numMainEffects() == 3, "3 main effects");
        assertTrue(model.numInteractions() == 1, "1 interaction");

        // Score at origin: all features = 0
        // pwl0(0)=0, pwl1(0)=0, pwl2(0)=0
        // interaction(0, 0): x1=0 in x1_grid=[0,0.5,1], x2=0 in x2_grid=[-1,0,1]
        //   -> z[0][1] = 0.1
        // total = 0.5 + 0 + 0 + 0 + 0.1 = 0.6
        assertEquals(0.6, model.scoreDocument(new double[]{0, 0, 0}), 1e-9, "score at origin");

        // Score at upper bounds
        // pwl0(1)=1.0, pwl1(2)=0.8, pwl2(1)=0.5
        // interaction(1.0, 2.0): x2 clamped to 1.0 -> z[2][2]=1.0
        // total = 0.5 + 1.0 + 0.8 + 0.5 + 1.0 = 3.8
        assertEquals(3.8, model.scoreDocument(new double[]{1, 2, 1}), 1e-9, "score at upper");
    }

    static void testModelBatch() throws Exception {
        InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel model = DistilledGamLoader.fromJson(is);
        is.close();

        double[][] features = {{0, 0, 0}, {1, 2, 1}};
        double[] scores = model.score(features);
        assertEquals(0.6, scores[0], 1e-9, "batch score[0]");
        assertEquals(3.8, scores[1], 1e-9, "batch score[1]");
    }

    static void testGamOnlyModel() {
        DistilledGamModel model = new DistilledGamModel(
                1.0,
                Arrays.asList(
                        new DistilledGamModel.MainEffect(0,
                                new PwlFunction(new double[]{0, 1}, new double[]{0, 2})),
                        new DistilledGamModel.MainEffect(1,
                                new PwlFunction(new double[]{0, 1}, new double[]{0, 1}))
                ),
                List.of()
        );
        // bias=1 + pwl0(0.5)=1.0 + pwl1(0.5)=0.5 = 2.5
        assertEquals(2.5, model.scoreDocument(new double[]{0.5, 0.5}), 1e-9, "gam only");
    }

    static void testJsonString() throws Exception {
        String json = "{\"bias\": 1.0, \"main_effects\": ["
                + "{\"feature\": 0, \"x\": [0.0, 1.0], \"y\": [0.0, 2.0]}"
                + "], \"interactions\": []}";
        DistilledGamModel m = DistilledGamLoader.fromJsonString(json);
        assertEquals(1.0, m.bias(), 1e-9, "json string bias");
        assertTrue(m.numMainEffects() == 1, "json string 1 main effect");
        assertEquals(2.0, m.scoreDocument(new double[]{0.5}), 1e-9, "json string score");
    }

    static void testScoreFinite() throws Exception {
        InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel model = DistilledGamLoader.fromJson(is);
        is.close();

        for (double v : new double[]{-1000, -1, 0, 0.5, 1, 1000}) {
            double score = model.scoreDocument(new double[]{v, v, v});
            assertTrue(Double.isFinite(score), "finite score for input " + v);
        }
    }

    // ── CompiledPwlFunction tests ──

    static void testCompiledMatchesPwl() {
        // Test all knot counts K=1..7 — compiled must match PwlFunction exactly
        double[][] xKnotsArr = {
                {5.0},                              // K=1
                {0, 1},                             // K=2
                {0, 0.5, 1},                        // K=3
                {0, 0.3, 0.7, 1},                   // K=4
                {0, 0.2, 0.4, 0.7, 1},              // K=5
                {0, 0.15, 0.3, 0.5, 0.75, 1},       // K=6
                {0, 0.1, 0.25, 0.4, 0.6, 0.8, 1},   // K=7 (fallback)
        };
        double[][] yKnotsArr = {
                {3.0},
                {0, 2},
                {0, 0.3, 1},
                {0, 0.5, 0.5, 2},
                {0, 0.1, 0.6, 0.8, 1.5},
                {0, 0.1, 0.3, 0.7, 0.9, 1.2},
                {0, 0.05, 0.2, 0.5, 0.7, 0.9, 1.0},
        };
        double[] testPoints = {-1, 0, 0.1, 0.25, 0.3, 0.5, 0.75, 0.9, 1.0, 2.0};

        for (int ki = 0; ki < xKnotsArr.length; ki++) {
            PwlFunction pwl = new PwlFunction(xKnotsArr[ki], yKnotsArr[ki]);
            CompiledPwlFunction compiled = CompiledPwlFunction.compile(pwl);

            for (double x : testPoints) {
                double expected = pwl.evaluate(x);
                double actual = compiled.evaluate(x);
                assertEquals(expected, actual, 1e-12,
                        "compiled K=" + xKnotsArr[ki].length + " at x=" + x);
            }
        }
    }

    static void testCompiledBulkAccumulate() {
        PwlFunction pwl = new PwlFunction(
                new double[]{0, 0.5, 1}, new double[]{0, 0.3, 1});
        CompiledPwlFunction compiled = CompiledPwlFunction.compile(pwl);

        double[] values = {-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5};
        double[] scores = new double[values.length];
        compiled.evaluateAndAccumulate(values, scores, values.length);

        for (int i = 0; i < values.length; i++) {
            assertEquals(pwl.evaluate(values[i]), scores[i], 1e-12,
                    "bulk accumulate at " + values[i]);
        }
    }

    static void testColumnMajorScoring() throws Exception {
        // Verify column-major model.score() matches row-major model.scoreDocument()
        InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel model = DistilledGamLoader.fromJson(is);
        is.close();

        java.util.Random rng = new java.util.Random(123);
        int listSize = 50;
        double[][] features = new double[listSize][3];
        for (int i = 0; i < listSize; i++) {
            for (int j = 0; j < 3; j++) {
                features[i][j] = rng.nextGaussian();
            }
        }

        double[] bulkScores = model.score(features);
        for (int i = 0; i < listSize; i++) {
            double singleScore = model.scoreDocument(features[i]);
            assertEquals(singleScore, bulkScores[i], 1e-9,
                    "column-major vs row-major doc " + i);
        }
    }

    static void testColumnarScoring() throws Exception {
        // Verify scoreColumnar() matches score() (row-major input)
        InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel model = DistilledGamLoader.fromJson(is);
        is.close();

        java.util.Random rng = new java.util.Random(456);
        int numDocs = 100;
        int numFeatures = 3;

        // Build row-major
        double[][] rowMajor = new double[numDocs][numFeatures];
        for (int i = 0; i < numDocs; i++)
            for (int j = 0; j < numFeatures; j++)
                rowMajor[i][j] = rng.nextGaussian();

        // Build column-major (SoA)
        double[][] columns = new double[numFeatures][numDocs];
        for (int i = 0; i < numDocs; i++)
            for (int j = 0; j < numFeatures; j++)
                columns[j][i] = rowMajor[i][j];

        double[] rowScores = model.score(rowMajor);
        double[] colScores = model.scoreColumnar(columns, numDocs);

        for (int i = 0; i < numDocs; i++) {
            assertEquals(rowScores[i], colScores[i], 1e-9,
                    "columnar vs row-major doc " + i);
        }
    }

    // ── Cross-validation with Python ──

    static void testMatchesPythonPwlPredict() throws Exception {
        // Verify the Java inference matches np.interp behavior
        // Linear function: knots at [0, 0.5, 1], values [0, 0.3, 1.0]
        PwlFunction f = new PwlFunction(
                new double[]{0, 0.5, 1.0},
                new double[]{0, 0.3, 1.0});

        // np.interp(0.25, [0, 0.5, 1], [0, 0.3, 1]) = 0.15
        assertEquals(0.15, f.evaluate(0.25), 1e-9, "matches np.interp at 0.25");
        // np.interp(0.75, ...) = 0.3 + 0.25/0.5 * 0.7 = 0.65
        assertEquals(0.65, f.evaluate(0.75), 1e-9, "matches np.interp at 0.75");
    }

    // ── ConcavePwlFunction tests ──

    static void testConcavePwlBasic() {
        // 4-segment concave PWL: slopes 2.0, 1.5, 1.0, 0.5 (decreasing)
        ConcavePwlFunction f = new ConcavePwlFunction(
                0.0,
                new double[]{0.0, 0.25, 0.5, 0.75},
                new double[]{0.25, 0.25, 0.25, 0.25},
                new double[]{2.0, 1.5, 1.0, 0.5},
                0.0, 1.0);

        // At x=0: intercept only = 0
        assertEquals(0.0, f.evaluate(0.0), 1e-9, "concave at 0");

        // At x=0.25: 2.0*0.25 = 0.5
        assertEquals(0.5, f.evaluate(0.25), 1e-9, "concave at 0.25");

        // At x=0.5: 2.0*0.25 + 1.5*0.25 = 0.5 + 0.375 = 0.875
        assertEquals(0.875, f.evaluate(0.5), 1e-9, "concave at 0.5");

        // At x=0.75: 0.875 + 1.0*0.25 = 1.125
        assertEquals(1.125, f.evaluate(0.75), 1e-9, "concave at 0.75");

        // At x=1.0: 1.125 + 0.5*0.25 = 1.25
        assertEquals(1.25, f.evaluate(1.0), 1e-9, "concave at 1.0");
    }

    static void testConcavePwlClamp() {
        ConcavePwlFunction f = new ConcavePwlFunction(
                0.1,
                new double[]{0.0, 0.5},
                new double[]{0.5, 0.5},
                new double[]{1.0, 0.4},
                0.0, 1.0);

        // Below xMin: clamp to 0 → intercept = 0.1
        assertEquals(0.1, f.evaluate(-1.0), 1e-9, "concave clamp below");

        // Above xMax: clamp to 1.0 → 0.1 + 1.0*0.5 + 0.4*0.5 = 0.1 + 0.5 + 0.2 = 0.8
        assertEquals(0.8, f.evaluate(2.0), 1e-9, "concave clamp above");
        assertEquals(0.8, f.evaluate(1.0), 1e-9, "concave at xMax");
    }

    static void testConcavePwlEarlyBreak() {
        // Verify the early-break optimization works correctly
        ConcavePwlFunction f = new ConcavePwlFunction(
                0.0,
                new double[]{0.0, 0.25, 0.5, 0.75},
                new double[]{0.25, 0.25, 0.25, 0.25},
                new double[]{4.0, 3.0, 2.0, 1.0},
                0.0, 1.0);

        // At x=0.1: only first segment active, 4.0 * 0.1 = 0.4
        assertEquals(0.4, f.evaluate(0.1), 1e-9, "concave early break");
    }

    static void testConcavePwlConcavity() {
        // Verify concavity: f(a+h) - f(a) >= f(b+h) - f(b) when b > a (diminishing returns)
        ConcavePwlFunction f = new ConcavePwlFunction(
                0.0,
                new double[]{0.0, 0.25, 0.5, 0.75},
                new double[]{0.25, 0.25, 0.25, 0.25},
                new double[]{2.0, 1.5, 1.0, 0.5},
                0.0, 1.0);

        double h = 0.1;
        for (double a = 0.0; a < 0.8; a += 0.1) {
            double b = a + 0.2;
            if (b + h > 1.0) break;
            double gainA = f.evaluate(a + h) - f.evaluate(a);
            double gainB = f.evaluate(b + h) - f.evaluate(b);
            assertTrue(gainA >= gainB - 1e-9,
                    "concavity: gain at " + a + " >= gain at " + b);
        }
    }

    // ── DefaultGroupwiseComputer tests ──

    static void testCategoryNovelty() {
        // 5 docs, feature col 0 = category
        // categories: [A=1, B=2, A=1, C=3, B=2]
        double[][] feats = {
                {1.0}, {2.0}, {1.0}, {3.0}, {2.0}
        };

        DefaultGroupwiseComputer.Spec spec = DefaultGroupwiseComputer.Spec.novelty("category_novelty", 0);
        DefaultGroupwiseComputer computer = new DefaultGroupwiseComputer(new DefaultGroupwiseComputer.Spec[]{spec});

        // Empty set: max novelty
        double[] r0 = computer.compute(0, new int[]{}, feats);
        assertEquals(1.0, r0[0], 1e-9, "novelty empty set");

        // S={0} (cat=1), candidate 1 (cat=2): 1 - 0/1 = 1.0
        double[] r1 = computer.compute(1, new int[]{0}, feats);
        assertEquals(1.0, r1[0], 1e-9, "novelty diff category");

        // S={0} (cat=1), candidate 2 (cat=1): 1 - 1/1 = 0.0
        double[] r2 = computer.compute(2, new int[]{0}, feats);
        assertEquals(0.0, r2[0], 1e-9, "novelty same category");

        // S={0, 1} (cat=1,2), candidate 4 (cat=2): 1 - 1/2 = 0.5
        double[] r3 = computer.compute(4, new int[]{0, 1}, feats);
        assertEquals(0.5, r3[0], 1e-9, "novelty partial match");
    }

    // ── SubmodularGamReranker tests ──

    static void testSubmodularRerankerBasic() throws Exception {
        // Load base model
        java.io.InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel baseModel = DistilledGamLoader.fromJson(is);
        is.close();

        // Create a simple diversity tower
        ConcavePwlFunction tower = new ConcavePwlFunction(
                0.0,
                new double[]{0.0, 0.5},
                new double[]{0.5, 0.5},
                new double[]{1.0, 0.4},
                0.0, 1.0);

        // Simple groupwise feature: returns 1.0 if empty set, else novelty on col 0
        GroupwiseFeatureComputer computer = new GroupwiseFeatureComputer() {
            @Override
            public double[] compute(int candIdx, int[] selIdx, double[][] feats) {
                if (selIdx.length == 0) return new double[]{1.0};
                double candVal = feats[candIdx][0];
                int matches = 0;
                for (int s : selIdx) {
                    if (Math.abs(feats[s][0] - candVal) < 0.01) matches++;
                }
                return new double[]{1.0 - (double) matches / selIdx.length};
            }

            @Override
            public int numFeatures() { return 1; }
        };

        SubmodularGamReranker reranker = new SubmodularGamReranker(
                baseModel, new ConcavePwlFunction[]{tower}, computer);

        // 5 docs with 3 features (to match base model)
        double[][] features = {
                {0.5, 0.5, 0.5},
                {0.5, 0.5, 0.5},   // same as doc 0
                {1.0, 1.0, 1.0},
                {0.0, 0.0, 0.0},
                {0.8, 0.8, 0.8}
        };

        int[] order = reranker.rerank(features, 5);
        assertTrue(order.length == 5, "reranker returns 5 items");

        // Check all indices are unique
        boolean[] seen = new boolean[5];
        for (int idx : order) {
            assertTrue(!seen[idx], "reranker index unique: " + idx);
            seen[idx] = true;
        }
    }

    static void testSubmodularMaxEvalBudget() throws Exception {
        java.io.InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel baseModel = DistilledGamLoader.fromJson(is);
        is.close();

        ConcavePwlFunction tower = new ConcavePwlFunction(
                0.0, new double[]{0.0, 0.5}, new double[]{0.5, 0.5},
                new double[]{1.0, 0.4}, 0.0, 1.0);

        // Counter to track evaluations
        final int[] evalCount = {0};
        GroupwiseFeatureComputer computer = new GroupwiseFeatureComputer() {
            @Override
            public double[] compute(int candIdx, int[] selIdx, double[][] feats) {
                evalCount[0]++;
                if (selIdx.length == 0) return new double[]{1.0};
                return new double[]{0.5}; // simple constant diversity
            }

            @Override
            public int numFeatures() { return 1; }
        };

        SubmodularGamReranker reranker = new SubmodularGamReranker(
                baseModel, new ConcavePwlFunction[]{tower}, computer);

        // 100 docs, select 10, max 5 evals per position
        java.util.Random rng = new java.util.Random(42);
        double[][] features = new double[100][3];
        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 3; j++)
                features[i][j] = rng.nextGaussian();

        evalCount[0] = 0;
        int[] order = reranker.rerank(features, 10, 5);
        assertTrue(order.length == 10, "budget reranker returns 10 items");
        // Max evals: 10 positions * 5 per pos = 50 (first position uses at most n)
        // But first pos with maxEvalsPerPos=5 only does 5 evals
        assertTrue(evalCount[0] <= 50, "eval budget respected: " + evalCount[0] + " <= 50");

        // Compare with unlimited budget
        evalCount[0] = 0;
        int[] orderUnlimited = reranker.rerank(features, 10);
        int unlimitedEvals = evalCount[0];
        // Unlimited should have more evals (at least first position evaluates all)
        assertTrue(unlimitedEvals >= 10, "unlimited evals >= 10: " + unlimitedEvals);
    }

    static void testSubmodularDiversityEffect() throws Exception {
        java.io.InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel baseModel = DistilledGamLoader.fromJson(is);
        is.close();

        // Strong diversity tower
        ConcavePwlFunction tower = new ConcavePwlFunction(
                0.0, new double[]{0.0, 0.5}, new double[]{0.5, 0.5},
                new double[]{10.0, 5.0}, 0.0, 1.0);

        // Feature col 0 = category. 3 categories: A, B, C
        GroupwiseFeatureComputer computer = new GroupwiseFeatureComputer() {
            @Override
            public double[] compute(int candIdx, int[] selIdx, double[][] feats) {
                if (selIdx.length == 0) return new double[]{1.0};
                double candCat = feats[candIdx][0];
                int matches = 0;
                for (int s : selIdx) {
                    if (Math.abs(feats[s][0] - candCat) < 0.01) matches++;
                }
                return new double[]{1.0 - (double) matches / selIdx.length};
            }

            @Override
            public int numFeatures() { return 1; }
        };

        SubmodularGamReranker reranker = new SubmodularGamReranker(
                baseModel, new ConcavePwlFunction[]{tower}, computer);

        // 6 docs: 2 of each category. All same base features except col 0 (category).
        double[][] features = {
                {1.0, 0.5, 0.5}, // cat A
                {2.0, 0.5, 0.5}, // cat B
                {3.0, 0.5, 0.5}, // cat C
                {1.0, 0.5, 0.5}, // cat A (duplicate)
                {2.0, 0.5, 0.5}, // cat B (duplicate)
                {3.0, 0.5, 0.5}, // cat C (duplicate)
        };

        int[] order = reranker.rerank(features, 6);

        // With strong diversity, first 3 items should be from different categories
        java.util.Set<Double> firstThreeCats = new java.util.HashSet<>();
        for (int i = 0; i < 3; i++) {
            firstThreeCats.add(features[order[i]][0]);
        }
        assertTrue(firstThreeCats.size() == 3,
                "diversity: first 3 picks cover 3 categories, got " + firstThreeCats.size());
    }

    static void testLoadSubmodularModel() throws Exception {
        java.io.InputStream is = InferenceTest.class.getResourceAsStream(
                "/test_submodular_model.json");
        DistilledGamLoader.SubmodularModelData data =
                DistilledGamLoader.loadSubmodular(is);
        is.close();

        assertTrue(data.baseModel.numMainEffects() == 3, "submod: 3 main effects");
        assertTrue(data.diversityTowers.length == 2, "submod: 2 diversity towers");

        // Verify tower 0 evaluation (4-segment, slopes 2.0, 1.5, 1.0, 0.5)
        assertEquals(0.0, data.diversityTowers[0].evaluate(0.0), 1e-9, "div tower 0 at 0");
        assertEquals(1.25, data.diversityTowers[0].evaluate(1.0), 1e-9, "div tower 0 at 1");

        // Verify tower 1 evaluation (intercept=0.1, 2-segment, slopes 1.0, 0.4)
        assertEquals(0.1, data.diversityTowers[1].evaluate(0.0), 1e-9, "div tower 1 at 0");
        // At 1.0: 0.1 + 1.0*0.5 + 0.4*0.5 = 0.8
        assertEquals(0.8, data.diversityTowers[1].evaluate(1.0), 1e-9, "div tower 1 at 1");
    }

    static void testRerankWithScores() throws Exception {
        java.io.InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel baseModel = DistilledGamLoader.fromJson(is);
        is.close();

        ConcavePwlFunction tower = new ConcavePwlFunction(
                0.0, new double[]{0.0, 0.5}, new double[]{0.5, 0.5},
                new double[]{1.0, 0.4}, 0.0, 1.0);

        GroupwiseFeatureComputer computer = new GroupwiseFeatureComputer() {
            @Override
            public double[] compute(int candIdx, int[] selIdx, double[][] feats) {
                return new double[]{selIdx.length == 0 ? 1.0 : 0.5};
            }
            @Override
            public int numFeatures() { return 1; }
        };

        SubmodularGamReranker reranker = new SubmodularGamReranker(
                baseModel, new ConcavePwlFunction[]{tower}, computer);

        double[][] features = {{0.5, 0.5, 0.5}, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}};
        SubmodularGamReranker.RerankResult result =
                reranker.rerankWithScores(features, 3, 0);

        assertTrue(result.indices.length == 3, "rerankWithScores returns 3 items");
        assertTrue(result.scores.length == 3, "rerankWithScores returns 3 scores");
        // First score should be highest (greedy picks best first)
        assertTrue(result.scores[0] >= result.scores[1] - 1e-9,
                "scores non-increasing: " + result.scores[0] + " >= " + result.scores[1]);
    }

    // ── Benchmark ──

    static void benchmarkReranker() throws Exception {
        java.io.InputStream is = InferenceTest.class.getResourceAsStream("/test_model.json");
        DistilledGamModel baseModel = DistilledGamLoader.fromJson(is);
        is.close();

        ConcavePwlFunction[] towers = {
                new ConcavePwlFunction(0.0, new double[]{0.0, 0.25, 0.5, 0.75},
                        new double[]{0.25, 0.25, 0.25, 0.25},
                        new double[]{2.0, 1.5, 1.0, 0.5}, 0.0, 1.0),
                new ConcavePwlFunction(0.1, new double[]{0.0, 0.5},
                        new double[]{0.5, 0.5},
                        new double[]{1.0, 0.4}, 0.0, 1.0),
        };

        GroupwiseFeatureComputer computer = new GroupwiseFeatureComputer() {
            @Override
            public double[] compute(int candIdx, int[] selIdx, double[][] feats) {
                if (selIdx.length == 0) return new double[]{1.0, 1.0};
                double cat = feats[candIdx][0];
                int m = 0;
                for (int s : selIdx) {
                    if (Math.abs(feats[s][0] - cat) < 0.01) m++;
                }
                double novelty = 1.0 - (double) m / selIdx.length;
                return new double[]{novelty, novelty * 0.8};
            }
            @Override
            public int numFeatures() { return 2; }
        };

        SubmodularGamReranker reranker = new SubmodularGamReranker(
                baseModel, towers, computer);

        System.out.println("\n--- Benchmarks: SubmodularGamReranker ---");

        for (int listSize : new int[]{100, 1000, 10_000}) {
            java.util.Random rng = new java.util.Random(42);
            double[][] features = new double[listSize][3];
            // Assign categories 0-9 to col 0
            for (int i = 0; i < listSize; i++) {
                features[i][0] = i % 10;
                features[i][1] = rng.nextGaussian();
                features[i][2] = rng.nextGaussian();
            }

            int k = 40;

            // Warmup
            for (int w = 0; w < 3; w++) reranker.rerank(features, k, 10);

            // Benchmark with max 10 evals
            int iterations = Math.max(10, 1000 / listSize);
            long start = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                reranker.rerank(features, k, 10);
            }
            long elapsed = System.nanoTime() - start;
            double usPerRerank = elapsed / 1000.0 / iterations;

            // Benchmark unlimited (Minoux lazy greedy)
            for (int w = 0; w < 3; w++) reranker.rerank(features, k);
            start = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                reranker.rerank(features, k);
            }
            long elapsedUnlimited = System.nanoTime() - start;
            double usPerRerankUnlimited = elapsedUnlimited / 1000.0 / iterations;

            System.out.printf("  %,6d docs, k=%d: budget=10 %8.1f us, lazy-greedy %8.1f us%n",
                    listSize, k, usPerRerank, usPerRerankUnlimited);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Running DistilledGamModel inference tests...\n");

        testPwlLinear();
        testPwlClamp();
        testPwlMultiSegment();
        testPwlBatch();
        testBilinearCorners();
        testBilinearCenter();
        testBilinearEdge();
        testBilinearClamp();
        testModelFromJson();
        testModelBatch();
        testGamOnlyModel();
        testJsonString();
        testScoreFinite();
        testCompiledMatchesPwl();
        testCompiledBulkAccumulate();
        testColumnMajorScoring();
        testColumnarScoring();
        testMatchesPythonPwlPredict();

        // Submodular tests
        testConcavePwlBasic();
        testConcavePwlClamp();
        testConcavePwlEarlyBreak();
        testConcavePwlConcavity();
        testCategoryNovelty();
        testSubmodularRerankerBasic();
        testSubmodularMaxEvalBudget();
        testSubmodularDiversityEffect();
        testLoadSubmodularModel();
        testRerankWithScores();

        System.out.println("\n" + passed + " passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        } else {
            System.out.println("ALL TESTS PASSED");
        }

        // Run benchmarks after tests pass
        benchmarkReranker();
    }
}
