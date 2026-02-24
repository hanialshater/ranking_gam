package com.rankinggam.inference;

import java.util.Collections;
import java.util.List;

/**
 * Inference engine for distilled GAM/GA2M ranking models.
 *
 * <p>Evaluates the scoring function:
 * <pre>
 *   score(x) = bias + sum_j pwl_j(x_j) + sum_k interp_k(x_{f1_k}, x_{f2_k})
 * </pre>
 *
 * <p>Models are exported from Python via {@code save_pwl_json()} and loaded here
 * via {@link DistilledGamLoader}. The inference is pure arithmetic (no ML framework
 * dependencies), suitable for low-latency ranking services.
 *
 * <p>Usage:
 * <pre>
 *   DistilledGamModel model = DistilledGamLoader.fromJson(path);
 *
 *   // Score a single document
 *   double score = model.scoreDocument(features);
 *
 *   // Score a list of candidate documents for one query
 *   double[] scores = model.score(featureMatrix);
 * </pre>
 */
public final class DistilledGamModel {

    private final double bias;
    private final List<MainEffect> mainEffects;
    private final List<Interaction> interactions;

    /**
     * A main-effect tower: one PWL function for feature index {@code featureIndex}.
     */
    public static final class MainEffect {
        private final int featureIndex;
        private final PwlFunction pwl;

        public MainEffect(int featureIndex, PwlFunction pwl) {
            this.featureIndex = featureIndex;
            this.pwl = pwl;
        }

        public int featureIndex() {
            return featureIndex;
        }

        public PwlFunction pwl() {
            return pwl;
        }
    }

    /**
     * A pairwise interaction tower: bilinear grid over (feature1, feature2).
     */
    public static final class Interaction {
        private final int feature1;
        private final int feature2;
        private final BilinearGridFunction grid;

        public Interaction(int feature1, int feature2, BilinearGridFunction grid) {
            this.feature1 = feature1;
            this.feature2 = feature2;
            this.grid = grid;
        }

        public int feature1() {
            return feature1;
        }

        public int feature2() {
            return feature2;
        }

        public BilinearGridFunction grid() {
            return grid;
        }
    }

    public DistilledGamModel(double bias, List<MainEffect> mainEffects, List<Interaction> interactions) {
        this.bias = bias;
        this.mainEffects = Collections.unmodifiableList(mainEffects);
        this.interactions = Collections.unmodifiableList(interactions);
    }

    /**
     * Score a single document.
     *
     * @param features feature values for one document, length >= max feature index used
     * @return relevance score
     */
    public double scoreDocument(double[] features) {
        double score = bias;

        for (MainEffect me : mainEffects) {
            score += me.pwl.evaluate(features[me.featureIndex]);
        }

        for (Interaction ia : interactions) {
            score += ia.grid.evaluate(features[ia.feature1], features[ia.feature2]);
        }

        return score;
    }

    /**
     * Score a list of candidate documents for one query.
     *
     * @param features [listSize][numFeatures] feature matrix
     * @return scores array of length listSize
     */
    public double[] score(double[][] features) {
        double[] scores = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            scores[i] = scoreDocument(features[i]);
        }
        return scores;
    }

    /**
     * Score a batch of queries, each with a list of candidate documents.
     *
     * @param batch [batchSize][listSize][numFeatures]
     * @return [batchSize][listSize] score matrix
     */
    public double[][] scoreBatch(double[][][] batch) {
        double[][] result = new double[batch.length][];
        for (int b = 0; b < batch.length; b++) {
            result[b] = score(batch[b]);
        }
        return result;
    }

    /** Global bias term. */
    public double bias() {
        return bias;
    }

    /** Number of main-effect (single-feature) towers. */
    public int numMainEffects() {
        return mainEffects.size();
    }

    /** Number of interaction (pairwise) towers. */
    public int numInteractions() {
        return interactions.size();
    }

    /** Unmodifiable list of main effects. */
    public List<MainEffect> mainEffects() {
        return mainEffects;
    }

    /** Unmodifiable list of interactions. */
    public List<Interaction> interactions() {
        return interactions;
    }
}
