package com.rankinggam.inference;

/**
 * Built-in groupwise feature computer supporting standard diversity feature types.
 *
 * <p>Supported types (matching Python's GroupwiseFeatureComputer):
 * <ul>
 *   <li>{@code category_novelty}: 1 - |same_category_in_S| / |S|
 *   <li>{@code brand_novelty}: 1 - |same_brand_in_S| / |S|
 *   <li>{@code visual_diversity}: 1 - max_cosine_similarity_to_S
 *   <li>{@code price_spread}: |price - mean_price_S| / (std_price_S + eps)
 *   <li>{@code attribute_coverage}: fraction of item's attributes not yet covered by S
 * </ul>
 *
 * <p>Configure via {@link Spec} array matching the model's diversity towers.
 */
public final class DefaultGroupwiseComputer implements GroupwiseFeatureComputer {

    /** Specification for a single groupwise feature. */
    public static final class Spec {
        final String type;
        final int column;       // for category_novelty, brand_novelty, price_spread
        final int[] columns;    // for visual_diversity, attribute_coverage
        final double xMax;

        private Spec(String type, int column, int[] columns, double xMax) {
            this.type = type;
            this.column = column;
            this.columns = columns;
            this.xMax = xMax;
        }

        /** Create a spec for category_novelty or brand_novelty. */
        public static Spec novelty(String type, int column) {
            return new Spec(type, column, null, 1.0);
        }

        /** Create a spec for visual_diversity. */
        public static Spec visualDiversity(int[] columns) {
            return new Spec("visual_diversity", -1, columns.clone(), 1.0);
        }

        /** Create a spec for price_spread. */
        public static Spec priceSpread(int column, double xMax) {
            return new Spec("price_spread", column, null, xMax);
        }

        /** Create a spec for attribute_coverage. */
        public static Spec attributeCoverage(int[] columns) {
            return new Spec("attribute_coverage", -1, columns.clone(), 1.0);
        }
    }

    private final Spec[] specs;
    private final double[] defaultValues; // values when S is empty

    public DefaultGroupwiseComputer(Spec[] specs, double[] defaultValues) {
        this.specs = specs;
        this.defaultValues = defaultValues;
    }

    /**
     * Create from specs, using xMax as default value for each feature (max diversity when S empty).
     */
    public DefaultGroupwiseComputer(Spec[] specs) {
        this.specs = specs;
        this.defaultValues = new double[specs.length];
        for (int i = 0; i < specs.length; i++) {
            defaultValues[i] = specs[i].xMax;
        }
    }

    @Override
    public int numFeatures() {
        return specs.length;
    }

    @Override
    public double[] compute(int candidateIdx, int[] selectedIndices, double[][] features) {
        double[] result = new double[specs.length];

        if (selectedIndices.length == 0) {
            System.arraycopy(defaultValues, 0, result, 0, specs.length);
            return result;
        }

        for (int j = 0; j < specs.length; j++) {
            result[j] = computeOne(specs[j], candidateIdx, selectedIndices, features);
        }
        return result;
    }

    private double computeOne(Spec spec, int candidateIdx, int[] sel, double[][] feats) {
        switch (spec.type) {
            case "category_novelty":
            case "brand_novelty":
                return computeNovelty(candidateIdx, sel, feats, spec.column);
            case "visual_diversity":
                return computeVisualDiversity(candidateIdx, sel, feats, spec.columns);
            case "price_spread":
                return computePriceSpread(candidateIdx, sel, feats, spec.column, spec.xMax);
            case "attribute_coverage":
                return computeAttributeCoverage(candidateIdx, sel, feats, spec.columns);
            default:
                return spec.xMax; // unknown type: return max
        }
    }

    private static double computeNovelty(int candIdx, int[] sel, double[][] feats, int col) {
        double candVal = feats[candIdx][col];
        int matches = 0;
        for (int s : sel) {
            if (feats[s][col] == candVal) matches++;
        }
        return 1.0 - (double) matches / sel.length;
    }

    private static double computeVisualDiversity(int candIdx, int[] sel, double[][] feats, int[] cols) {
        double candNormSq = 0;
        for (int c : cols) {
            double v = feats[candIdx][c];
            candNormSq += v * v;
        }
        double candNorm = Math.sqrt(candNormSq);
        if (candNorm < 1e-10) return 1.0;

        double maxSim = -1;
        for (int s : sel) {
            double dot = 0, selNormSq = 0;
            for (int c : cols) {
                double cv = feats[candIdx][c], sv = feats[s][c];
                dot += cv * sv;
                selNormSq += sv * sv;
            }
            double selNorm = Math.sqrt(selNormSq);
            if (selNorm < 1e-10) continue;
            double sim = dot / (candNorm * selNorm);
            if (sim > maxSim) maxSim = sim;
        }
        return maxSim < 0 ? 1.0 : 1.0 - maxSim;
    }

    private static double computePriceSpread(int candIdx, int[] sel, double[][] feats,
                                              int col, double xMax) {
        double mean = 0;
        for (int s : sel) mean += feats[s][col];
        mean /= sel.length;

        double variance = 0;
        if (sel.length > 1) {
            for (int s : sel) {
                double d = feats[s][col] - mean;
                variance += d * d;
            }
            variance /= sel.length;
        } else {
            variance = 1.0;
        }
        double std = Math.sqrt(variance);
        if (std < 1e-6) std = 1e-6;

        double spread = Math.abs(feats[candIdx][col] - mean) / std;
        return Math.min(spread, xMax);
    }

    private static double computeAttributeCoverage(int candIdx, int[] sel, double[][] feats,
                                                    int[] cols) {
        int candTotal = 0, candNew = 0;
        for (int c : cols) {
            boolean candHas = feats[candIdx][c] > 0;
            if (!candHas) continue;
            candTotal++;
            boolean covered = false;
            for (int s : sel) {
                if (feats[s][c] > 0) { covered = true; break; }
            }
            if (!covered) candNew++;
        }
        return candTotal > 0 ? (double) candNew / candTotal : 0.0;
    }
}
