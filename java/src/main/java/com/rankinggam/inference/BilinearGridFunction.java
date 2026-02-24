package com.rankinggam.inference;

/**
 * Bilinear interpolation on a 2D grid for GA2M interaction effects.
 *
 * <p>Stores a grid of z-values at (x1, x2) coordinates and performs bilinear
 * interpolation for arbitrary query points. Matches the behavior of
 * {@code scipy.interpolate.RegularGridInterpolator} with method="linear"
 * and extrapolation clamped to grid boundaries.
 */
public final class BilinearGridFunction {

    private final double[] x1Grid;
    private final double[] x2Grid;
    private final double[][] z; // z[i][j] = value at (x1Grid[i], x2Grid[j])

    /**
     * @param x1Grid sorted x1 coordinates (first axis)
     * @param x2Grid sorted x2 coordinates (second axis)
     * @param z      z[i][j] = function value at (x1Grid[i], x2Grid[j])
     */
    public BilinearGridFunction(double[] x1Grid, double[] x2Grid, double[][] z) {
        if (x1Grid.length < 2 || x2Grid.length < 2) {
            throw new IllegalArgumentException("Grid must have at least 2 points per axis");
        }
        if (z.length != x1Grid.length) {
            throw new IllegalArgumentException(
                    "z rows (" + z.length + ") must match x1Grid length (" + x1Grid.length + ")");
        }
        this.x1Grid = x1Grid.clone();
        this.x2Grid = x2Grid.clone();
        this.z = new double[z.length][];
        for (int i = 0; i < z.length; i++) {
            if (z[i].length != x2Grid.length) {
                throw new IllegalArgumentException(
                        "z[" + i + "] length (" + z[i].length
                                + ") must match x2Grid length (" + x2Grid.length + ")");
            }
            this.z[i] = z[i].clone();
        }
    }

    /**
     * Evaluate the bilinear interpolation at a single (x1, x2) point.
     * Out-of-range values are clamped to grid boundaries.
     */
    public double evaluate(double x1, double x2) {
        // Clamp to grid
        x1 = clamp(x1, x1Grid[0], x1Grid[x1Grid.length - 1]);
        x2 = clamp(x2, x2Grid[0], x2Grid[x2Grid.length - 1]);

        // Find cell indices
        int i = findSegment(x1Grid, x1);
        int j = findSegment(x2Grid, x2);

        // Interpolation weights
        double s = (x1Grid[i + 1] - x1Grid[i]) < 1e-12 ? 0.0
                : (x1 - x1Grid[i]) / (x1Grid[i + 1] - x1Grid[i]);
        double t = (x2Grid[j + 1] - x2Grid[j]) < 1e-12 ? 0.0
                : (x2 - x2Grid[j]) / (x2Grid[j + 1] - x2Grid[j]);

        // Bilinear interpolation
        double v00 = z[i][j];
        double v10 = z[i + 1][j];
        double v01 = z[i][j + 1];
        double v11 = z[i + 1][j + 1];

        return (1 - s) * (1 - t) * v00
                + s * (1 - t) * v10
                + (1 - s) * t * v01
                + s * t * v11;
    }

    /**
     * Binary search for the largest index k such that grid[k] <= x and k < grid.length - 1.
     */
    private static int findSegment(double[] grid, double x) {
        int lo = 0;
        int hi = grid.length - 2; // max valid segment start
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (grid[mid] <= x) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    private static double clamp(double x, double min, double max) {
        return Math.max(min, Math.min(max, x));
    }

    /**
     * Bulk evaluate: add bilinear contributions into a scores accumulator.
     * Column-major pattern — call once per interaction across all documents.
     *
     * @param x1Values  feature values for axis 1 (length >= count)
     * @param x2Values  feature values for axis 2 (length >= count)
     * @param scores    accumulator; contributions are ADDED in place
     * @param count     number of elements to process
     */
    void evaluateAndAccumulate(double[] x1Values, double[] x2Values,
                                double[] scores, int count) {
        final double x1Lo = x1Grid[0], x1Hi = x1Grid[x1Grid.length - 1];
        final double x2Lo = x2Grid[0], x2Hi = x2Grid[x2Grid.length - 1];

        for (int idx = 0; idx < count; idx++) {
            double x1 = x1Values[idx];
            double x2 = x2Values[idx];

            // Clamp
            if (x1 < x1Lo) x1 = x1Lo;
            else if (x1 > x1Hi) x1 = x1Hi;
            if (x2 < x2Lo) x2 = x2Lo;
            else if (x2 > x2Hi) x2 = x2Hi;

            // Find cell
            int i = findSegment(x1Grid, x1);
            int j = findSegment(x2Grid, x2);

            // Weights
            double dx1 = x1Grid[i + 1] - x1Grid[i];
            double s = dx1 < 1e-12 ? 0.0 : (x1 - x1Grid[i]) / dx1;
            double dx2 = x2Grid[j + 1] - x2Grid[j];
            double t = dx2 < 1e-12 ? 0.0 : (x2 - x2Grid[j]) / dx2;

            // Bilinear
            scores[idx] += (1 - s) * ((1 - t) * z[i][j] + t * z[i][j + 1])
                         + s * ((1 - t) * z[i + 1][j] + t * z[i + 1][j + 1]);
        }
    }

    /** Feature indices for this interaction (e.g., [3, 7]). Stored externally in the model. */
    public int x1Size() {
        return x1Grid.length;
    }

    public int x2Size() {
        return x2Grid.length;
    }
}
