package com.rankinggam.inference;

import java.util.Arrays;
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
 * <p>Performance: PWL lookups are compiled to flat if/else chains at load time
 * (no loops or binary search for K<=6 knots). Bulk scoring uses column-major
 * evaluation (feature-by-feature across all docs) for better cache utilization.
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

    // Compiled structures for bulk scoring
    private final CompiledPwlFunction[] compiledPwl;
    private final int[] pwlFeatureIndices;
    private final BilinearGridFunction[] interactionGrids;
    private final CompiledBilinearGridFunction[] compiledGrids;
    private final int[] interactionF1;
    private final int[] interactionF2;

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

        // Pre-compile PWL functions into if/else evaluators
        this.compiledPwl = new CompiledPwlFunction[mainEffects.size()];
        this.pwlFeatureIndices = new int[mainEffects.size()];
        for (int j = 0; j < mainEffects.size(); j++) {
            MainEffect me = mainEffects.get(j);
            this.compiledPwl[j] = CompiledPwlFunction.compile(me.pwl);
            this.pwlFeatureIndices[j] = me.featureIndex;
        }

        // Pre-extract and compile interaction grids
        this.interactionGrids = new BilinearGridFunction[interactions.size()];
        this.compiledGrids = new CompiledBilinearGridFunction[interactions.size()];
        this.interactionF1 = new int[interactions.size()];
        this.interactionF2 = new int[interactions.size()];
        for (int k = 0; k < interactions.size(); k++) {
            Interaction ia = interactions.get(k);
            this.interactionGrids[k] = ia.grid;
            this.compiledGrids[k] = CompiledBilinearGridFunction.compile(ia.grid);
            this.interactionF1[k] = ia.feature1;
            this.interactionF2[k] = ia.feature2;
        }
    }

    /**
     * Score a single document.
     *
     * @param features feature values for one document, length >= max feature index used
     * @return relevance score
     */
    public double scoreDocument(double[] features) {
        double score = bias;

        for (int j = 0; j < compiledPwl.length; j++) {
            score += compiledPwl[j].evaluate(features[pwlFeatureIndices[j]]);
        }

        for (int k = 0; k < compiledGrids.length; k++) {
            score += compiledGrids[k].evaluate(
                    features[interactionF1[k]], features[interactionF2[k]]);
        }

        return score;
    }

    /**
     * Score a list of candidate documents for one query.
     *
     * <p>Uses column-major evaluation: iterates feature-by-feature across all
     * documents, using compiled if/else evaluators and bulk accumulation.
     * This gives much better performance for large lists (1k-10k docs).
     *
     * @param features [listSize][numFeatures] feature matrix
     * @return scores array of length listSize
     */
    public double[] score(double[][] features) {
        final int listSize = features.length;
        if (listSize == 0) return new double[0];

        double[] scores = new double[listSize];
        Arrays.fill(scores, bias);

        // Column-major main effects: extract one feature column, evaluate across all docs
        double[] column = new double[listSize];

        for (int j = 0; j < compiledPwl.length; j++) {
            final int fi = pwlFeatureIndices[j];
            for (int i = 0; i < listSize; i++) {
                column[i] = features[i][fi];
            }
            compiledPwl[j].evaluateAndAccumulate(column, scores, listSize);
        }

        // Column-major interactions (compiled)
        if (compiledGrids.length > 0) {
            double[] col2 = new double[listSize];
            for (int k = 0; k < compiledGrids.length; k++) {
                final int f1 = interactionF1[k], f2 = interactionF2[k];
                for (int i = 0; i < listSize; i++) {
                    column[i] = features[i][f1];
                    col2[i] = features[i][f2];
                }
                compiledGrids[k].evaluateAndAccumulate(column, col2, scores, listSize);
            }
        }

        return scores;
    }

    /**
     * Score documents from column-major (Structure-of-Arrays) feature layout.
     *
     * <p>This is the fastest scoring path. When your ranking service already
     * stores features per-column (e.g. one array per feature across all docs),
     * use this method to skip the row-to-column transpose entirely.
     *
     * <p>Only columns referenced by the model's main effects and interactions
     * need to be present. The {@code columns} map is indexed by feature index.
     *
     * @param columns  feature columns: columns[featureIndex] = double[numDocs]
     * @param numDocs  number of documents to score
     * @return scores array of length numDocs
     */
    public double[] scoreColumnar(double[][] columns, int numDocs) {
        if (numDocs == 0) return new double[0];

        double[] scores = new double[numDocs];
        Arrays.fill(scores, bias);

        // Main effects — columns already in the right layout, zero-copy
        for (int j = 0; j < compiledPwl.length; j++) {
            compiledPwl[j].evaluateAndAccumulate(
                    columns[pwlFeatureIndices[j]], scores, numDocs);
        }

        // Interactions (compiled)
        for (int k = 0; k < compiledGrids.length; k++) {
            compiledGrids[k].evaluateAndAccumulate(
                    columns[interactionF1[k]], columns[interactionF2[k]],
                    scores, numDocs);
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
