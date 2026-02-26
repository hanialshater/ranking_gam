package com.rankgam.inference;

import java.util.List;

/**
 * Data classes representing a distilled PWL (piecewise-linear) GAM model.
 *
 * Mirrors the Python dict produced by {@code distill_to_pwl()} / {@code distill_context_model()}.
 */
public final class PwlModel {

    /** Bias (intercept) added to every document score. */
    public final double bias;

    /** One entry per feature – sorted by feature index. */
    public final List<MainEffect> mainEffects;

    /** Pairwise interaction surfaces (may be empty for pure GAM). */
    public final List<Interaction> interactions;

    public PwlModel(double bias, List<MainEffect> mainEffects, List<Interaction> interactions) {
        this.bias = bias;
        this.mainEffects = mainEffects;
        this.interactions = interactions;
    }

    // ------------------------------------------------------------------ //
    //  Main effect: 1-D piecewise-linear function for a single feature    //
    // ------------------------------------------------------------------ //
    public static final class MainEffect {
        /** Feature index in the input vector. */
        public final int feature;

        /** Sorted knot x-coordinates. */
        public final double[] xKnots;

        /** Corresponding knot y-values. */
        public final double[] yKnots;

        public MainEffect(int feature, double[] xKnots, double[] yKnots) {
            if (xKnots.length != yKnots.length || xKnots.length < 2) {
                throw new IllegalArgumentException(
                        "xKnots and yKnots must have equal length >= 2, got " + xKnots.length);
            }
            this.feature = feature;
            this.xKnots = xKnots;
            this.yKnots = yKnots;
        }
    }

    // ------------------------------------------------------------------ //
    //  Interaction: 2-D bilinear grid for a feature pair                  //
    // ------------------------------------------------------------------ //
    public static final class Interaction {
        /** The two feature indices. */
        public final int feature1;
        public final int feature2;

        /** Grid coordinates along each axis (sorted). */
        public final double[] x1Grid;
        public final double[] x2Grid;

        /** 2-D value grid, row-major: z[i][j] = f(x1Grid[i], x2Grid[j]). */
        public final double[][] z;

        public Interaction(int feature1, int feature2,
                           double[] x1Grid, double[] x2Grid, double[][] z) {
            this.feature1 = feature1;
            this.feature2 = feature2;
            this.x1Grid = x1Grid;
            this.x2Grid = x2Grid;
            this.z = z;
        }
    }
}
