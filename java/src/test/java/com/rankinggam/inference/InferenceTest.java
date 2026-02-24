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
        testMatchesPythonPwlPredict();

        System.out.println("\n" + passed + " passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        } else {
            System.out.println("ALL TESTS PASSED");
        }
    }
}
