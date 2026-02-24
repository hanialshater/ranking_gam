package com.rankinggam.inference;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DistilledGamModelTest {

    private static DistilledGamModel model;

    @BeforeAll
    static void loadModel() throws IOException {
        try (InputStream is = DistilledGamModelTest.class.getResourceAsStream("/test_model.json")) {
            assertNotNull(is, "test_model.json not found on classpath");
            model = DistilledGamLoader.fromJson(is);
        }
    }

    @Test
    void testModelStructure() {
        assertEquals(0.5, model.bias(), 1e-9);
        assertEquals(3, model.numMainEffects());
        assertEquals(1, model.numInteractions());
    }

    @Test
    void testMainEffectIndices() {
        List<DistilledGamModel.MainEffect> effects = model.mainEffects();
        assertEquals(0, effects.get(0).featureIndex());
        assertEquals(1, effects.get(1).featureIndex());
        assertEquals(2, effects.get(2).featureIndex());
    }

    @Test
    void testInteractionFeatures() {
        DistilledGamModel.Interaction ia = model.interactions().get(0);
        assertEquals(0, ia.feature1());
        assertEquals(1, ia.feature2());
    }

    @Test
    void testScoreDocument() {
        // Score with features at known knot points
        // feature[0]=0.0 -> pwl0=0.0
        // feature[1]=0.0 -> pwl1=0.0
        // feature[2]=0.0 -> pwl2=0.0
        // interaction(0.0, 0.0) -> grid(0,0) = 0.1 (x1=0, x2=0 which maps to grid x2=0.0)
        // Wait, x2_grid=[-1, 0, 1], so x2=0.0 is at grid index 1
        // interaction(x1=0.0, x2=0.0) -> z[0][1] = 0.1
        // total = bias + 0 + 0 + 0 + 0.1 = 0.6
        double score = model.scoreDocument(new double[]{0.0, 0.0, 0.0});
        assertEquals(0.6, score, 1e-9);
    }

    @Test
    void testScoreDocumentAtEndpoints() {
        // feature[0]=1.0 -> pwl0=1.0
        // feature[1]=2.0 -> pwl1=0.8
        // feature[2]=1.0 -> pwl2=0.5
        // interaction(1.0, 2.0) -> x2 clamped to 1.0 -> z[2][2] = 1.0
        // total = 0.5 + 1.0 + 0.8 + 0.5 + 1.0 = 3.8
        double score = model.scoreDocument(new double[]{1.0, 2.0, 1.0});
        assertEquals(3.8, score, 1e-9);
    }

    @Test
    void testScoreBatch() {
        double[][] features = {
                {0.0, 0.0, 0.0},
                {1.0, 2.0, 1.0}
        };
        double[] scores = model.score(features);
        assertEquals(2, scores.length);
        assertEquals(0.6, scores[0], 1e-9);
        assertEquals(3.8, scores[1], 1e-9);
    }

    @Test
    void testScoreBatch3D() {
        double[][][] batch = {
                {{0.0, 0.0, 0.0}, {1.0, 2.0, 1.0}},
                {{0.5, 0.0, 0.5}}
        };
        double[][] scores = model.scoreBatch(batch);
        assertEquals(2, scores.length);
        assertEquals(2, scores[0].length);
        assertEquals(1, scores[1].length);
    }

    @Test
    void testGamOnlyModel() {
        // Model with no interactions
        DistilledGamModel gamOnly = new DistilledGamModel(
                1.0,
                Arrays.asList(
                        new DistilledGamModel.MainEffect(0,
                                new PwlFunction(new double[]{0.0, 1.0}, new double[]{0.0, 2.0})),
                        new DistilledGamModel.MainEffect(1,
                                new PwlFunction(new double[]{0.0, 1.0}, new double[]{0.0, 1.0}))
                ),
                List.of()
        );

        assertEquals(0, gamOnly.numInteractions());
        // bias=1 + pwl0(0.5)=1.0 + pwl1(0.5)=0.5 = 2.5
        assertEquals(2.5, gamOnly.scoreDocument(new double[]{0.5, 0.5}), 1e-9);
    }

    @Test
    void testLoadFromJsonString() throws IOException {
        String json = "{\"bias\": 1.0, \"main_effects\": ["
                + "{\"feature\": 0, \"x\": [0.0, 1.0], \"y\": [0.0, 2.0]}"
                + "], \"interactions\": []}";
        DistilledGamModel m = DistilledGamLoader.fromJsonString(json);
        assertEquals(1.0, m.bias(), 1e-9);
        assertEquals(1, m.numMainEffects());
        assertEquals(0, m.numInteractions());
        // bias=1 + pwl(0.5)=1.0 = 2.0
        assertEquals(2.0, m.scoreDocument(new double[]{0.5}), 1e-9);
    }

    @Test
    void testScoreIsFinite() {
        // Ensure no NaN/Inf for edge cases
        double[] edgeCases = {-1000, -1, 0, 0.5, 1, 1000};
        for (double v : edgeCases) {
            double score = model.scoreDocument(new double[]{v, v, v});
            assertTrue(Double.isFinite(score), "Score should be finite for input " + v);
        }
    }
}
