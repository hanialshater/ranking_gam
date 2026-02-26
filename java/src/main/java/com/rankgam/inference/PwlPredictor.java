package com.rankgam.inference;

import java.util.Arrays;
import java.util.List;

/**
 * Normal (generic, runtime-interpreted) PWL inference.
 *
 * <p>This is the <b>unoptimized baseline</b> implementation. It reads knot
 * arrays at runtime and performs standard linear interpolation for every
 * feature × every document. Use this to measure baseline latency and compare
 * against the compiled/code-generated variant.
 *
 * <h3>Inference algorithm (per document)</h3>
 * <pre>
 *   score = bias
 *         + Σ_j  interp1d(x[j], mainEffect_j)       // main effects
 *         + Σ_k  interp2d(x[f1], x[f2], interaction_k)  // interactions (GA2M)
 * </pre>
 *
 * <p>Thread-safe: all state is final and the predict methods are pure functions.
 */
public final class PwlPredictor {

    private final PwlModel model;

    public PwlPredictor(PwlModel model) {
        this.model = model;
    }

    // ------------------------------------------------------------------ //
    //  Batch predict: features[batch][listSize][numFeatures] -> [batch][listSize]
    // ------------------------------------------------------------------ //

    /**
     * Score a full batch of queries.
     *
     * @param features [batchSize][listSize][numFeatures]
     * @return [batchSize][listSize] relevance scores
     */
    public double[][] predict(double[][][] features) {
        final int batchSize = features.length;
        final double[][] scores = new double[batchSize][];
        for (int b = 0; b < batchSize; b++) {
            scores[b] = predictQuery(features[b]);
        }
        return scores;
    }

    // ------------------------------------------------------------------ //
    //  Single-query predict: features[listSize][numFeatures] -> [listSize]
    // ------------------------------------------------------------------ //

    /**
     * Score all documents in a single query.
     *
     * @param docs [listSize][numFeatures]
     * @return [listSize] relevance scores
     */
    public double[] predictQuery(double[][] docs) {
        final int listSize = docs.length;
        final double[] scores = new double[listSize];
        Arrays.fill(scores, model.bias);

        final List<PwlModel.MainEffect> mains = model.mainEffects;
        final List<PwlModel.Interaction> inters = model.interactions;

        // Main effects
        for (int m = 0; m < mains.size(); m++) {
            final PwlModel.MainEffect me = mains.get(m);
            final int feat = me.feature;
            final double[] xk = me.xKnots;
            final double[] yk = me.yKnots;
            for (int d = 0; d < listSize; d++) {
                scores[d] += interp1d(docs[d][feat], xk, yk);
            }
        }

        // Interactions (GA2M)
        for (int k = 0; k < inters.size(); k++) {
            final PwlModel.Interaction ia = inters.get(k);
            for (int d = 0; d < listSize; d++) {
                scores[d] += interp2d(
                        docs[d][ia.feature1], docs[d][ia.feature2],
                        ia.x1Grid, ia.x2Grid, ia.z);
            }
        }

        return scores;
    }

    // ------------------------------------------------------------------ //
    //  Single-document predict
    // ------------------------------------------------------------------ //

    /**
     * Score a single document.
     *
     * @param features [numFeatures]
     * @return scalar relevance score
     */
    public double predictSingle(double[] features) {
        double score = model.bias;

        for (int m = 0; m < model.mainEffects.size(); m++) {
            final PwlModel.MainEffect me = model.mainEffects.get(m);
            score += interp1d(features[me.feature], me.xKnots, me.yKnots);
        }

        for (int k = 0; k < model.interactions.size(); k++) {
            final PwlModel.Interaction ia = model.interactions.get(k);
            score += interp2d(
                    features[ia.feature1], features[ia.feature2],
                    ia.x1Grid, ia.x2Grid, ia.z);
        }

        return score;
    }

    // ================================================================== //
    //  1-D piecewise-linear interpolation (equivalent to np.interp)       //
    // ================================================================== //

    /**
     * Standard linear interpolation with clamp-to-boundary.
     * Equivalent to {@code numpy.interp(x, xKnots, yKnots)}.
     */
    static double interp1d(double x, double[] xKnots, double[] yKnots) {
        final int n = xKnots.length;

        // Clamp below / above
        if (x <= xKnots[0]) return yKnots[0];
        if (x >= xKnots[n - 1]) return yKnots[n - 1];

        // Binary search for the enclosing segment
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (xKnots[mid] <= x) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation within [lo, hi]
        double span = xKnots[hi] - xKnots[lo];
        if (span < 1e-12) return yKnots[lo];
        double t = (x - xKnots[lo]) / span;
        return yKnots[lo] + t * (yKnots[hi] - yKnots[lo]);
    }

    // ================================================================== //
    //  2-D bilinear interpolation on a regular grid                       //
    // ================================================================== //

    /**
     * Bilinear interpolation on a regular 2-D grid with clamp-to-boundary.
     * Equivalent to {@code scipy.interpolate.RegularGridInterpolator(..., method='linear')}.
     */
    static double interp2d(double x1, double x2,
                           double[] x1Grid, double[] x2Grid, double[][] z) {
        // Clamp inputs to grid bounds
        x1 = Math.max(x1Grid[0], Math.min(x1, x1Grid[x1Grid.length - 1]));
        x2 = Math.max(x2Grid[0], Math.min(x2, x2Grid[x2Grid.length - 1]));

        // Find enclosing cell along axis 1
        int i0 = lowerIndex(x1Grid, x1);
        int i1 = Math.min(i0 + 1, x1Grid.length - 1);
        double s1 = (i0 == i1) ? 0.0 : (x1 - x1Grid[i0]) / (x1Grid[i1] - x1Grid[i0]);

        // Find enclosing cell along axis 2
        int j0 = lowerIndex(x2Grid, x2);
        int j1 = Math.min(j0 + 1, x2Grid.length - 1);
        double s2 = (j0 == j1) ? 0.0 : (x2 - x2Grid[j0]) / (x2Grid[j1] - x2Grid[j0]);

        // Bilinear blend
        double v00 = z[i0][j0];
        double v01 = z[i0][j1];
        double v10 = z[i1][j0];
        double v11 = z[i1][j1];
        return v00 * (1 - s1) * (1 - s2)
             + v01 * (1 - s1) * s2
             + v10 * s1 * (1 - s2)
             + v11 * s1 * s2;
    }

    /** Binary search returning the largest index i such that grid[i] <= x. */
    private static int lowerIndex(double[] grid, double x) {
        int lo = 0, hi = grid.length - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (grid[mid] <= x) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
}
