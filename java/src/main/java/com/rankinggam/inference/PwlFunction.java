package com.rankinggam.inference;

/**
 * Piecewise-linear function for distilled GAM tower inference.
 *
 * <p>Evaluates f(x) by linear interpolation between knots.
 * Matches the behavior of {@code np.interp} in the Python distillation code.
 *
 * <p>Optimized for typical GAM distillation (K=3-5 knots):
 * <ul>
 *   <li>Slopes precomputed at construction time (no division at eval)
 *   <li>Linear scan instead of binary search for small K
 *   <li>Bulk {@link #evaluateAndAccumulate} for column-major scoring
 * </ul>
 */
public final class PwlFunction {

    final double[] xKnots;
    final double[] yKnots;
    final double[] slopes;  // slopes[k] = (y[k+1]-y[k]) / (x[k+1]-x[k])
    final int n;

    /**
     * @param xKnots sorted x-coordinates of the knots
     * @param yKnots corresponding y-coordinates
     * @throws IllegalArgumentException if arrays are empty or different lengths
     */
    public PwlFunction(double[] xKnots, double[] yKnots) {
        if (xKnots.length == 0 || xKnots.length != yKnots.length) {
            throw new IllegalArgumentException(
                    "xKnots and yKnots must be non-empty and same length, got "
                            + xKnots.length + " and " + yKnots.length);
        }
        this.n = xKnots.length;
        this.xKnots = xKnots.clone();
        this.yKnots = yKnots.clone();
        this.slopes = new double[Math.max(n - 1, 1)];
        for (int k = 0; k < n - 1; k++) {
            double span = xKnots[k + 1] - xKnots[k];
            slopes[k] = span < 1e-12 ? 0.0 : (yKnots[k + 1] - yKnots[k]) / span;
        }
    }

    /**
     * Evaluate the piecewise-linear function at a single point.
     */
    public double evaluate(double x) {
        if (x <= xKnots[0]) return yKnots[0];
        if (x >= xKnots[n - 1]) return yKnots[n - 1];
        int k = findSegment(x);
        return yKnots[k] + slopes[k] * (x - xKnots[k]);
    }

    /**
     * Bulk evaluate: add PWL contributions into a scores accumulator.
     * Column-major pattern — call once per feature across all documents.
     *
     * @param values  input feature values (length >= count)
     * @param scores  accumulator; contributions are ADDED in place
     * @param count   number of elements to process
     */
    void evaluateAndAccumulate(double[] values, double[] scores, int count) {
        final double x0 = xKnots[0];
        final double xEnd = xKnots[n - 1];
        final double y0 = yKnots[0];
        final double yEnd = yKnots[n - 1];

        if (n <= 6) {
            // Linear scan — fewer branch mispredictions for small K
            for (int i = 0; i < count; i++) {
                double x = values[i];
                if (x <= x0) { scores[i] += y0; continue; }
                if (x >= xEnd) { scores[i] += yEnd; continue; }
                int k = 0;
                while (k < n - 2 && xKnots[k + 1] <= x) k++;
                scores[i] += yKnots[k] + slopes[k] * (x - xKnots[k]);
            }
        } else {
            for (int i = 0; i < count; i++) {
                double x = values[i];
                if (x <= x0) { scores[i] += y0; continue; }
                if (x >= xEnd) { scores[i] += yEnd; continue; }
                int k = findSegment(x);
                scores[i] += yKnots[k] + slopes[k] * (x - xKnots[k]);
            }
        }
    }

    private int findSegment(double x) {
        int lo = 0, hi = n - 1;
        while (lo < hi - 1) {
            int mid = (lo + hi) >>> 1;
            if (xKnots[mid] <= x) lo = mid; else hi = mid;
        }
        return lo;
    }

    /** Evaluate the function at multiple points. */
    public double[] evaluate(double[] xs) {
        double[] result = new double[xs.length];
        for (int i = 0; i < xs.length; i++) {
            result[i] = evaluate(xs[i]);
        }
        return result;
    }

    public int numKnots() { return n; }
    public double[] getXKnots() { return xKnots.clone(); }
    public double[] getYKnots() { return yKnots.clone(); }
}
