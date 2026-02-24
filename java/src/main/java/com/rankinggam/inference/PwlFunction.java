package com.rankinggam.inference;

/**
 * Piecewise-linear function for distilled GAM tower inference.
 *
 * <p>Evaluates f(x) by linear interpolation between knots.
 * Matches the behavior of {@code np.interp} in the Python distillation code.
 *
 * <p>Knots must be sorted in ascending x order. Values outside the knot range
 * are clamped to the nearest endpoint value.
 */
public final class PwlFunction {

    private final double[] xKnots;
    private final double[] yKnots;

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
        this.xKnots = xKnots.clone();
        this.yKnots = yKnots.clone();
    }

    /**
     * Evaluate the piecewise-linear function at a single point.
     *
     * <p>Uses binary search to find the segment, then linear interpolation.
     * Clamps to endpoint values for out-of-range inputs.
     */
    public double evaluate(double x) {
        int n = xKnots.length;

        // Clamp to endpoints
        if (x <= xKnots[0]) {
            return yKnots[0];
        }
        if (x >= xKnots[n - 1]) {
            return yKnots[n - 1];
        }

        // Binary search for the segment: find largest k such that xKnots[k] <= x
        int lo = 0;
        int hi = n - 1;
        while (lo < hi - 1) {
            int mid = (lo + hi) >>> 1;
            if (xKnots[mid] <= x) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation within segment [lo, lo+1]
        double span = xKnots[lo + 1] - xKnots[lo];
        if (span < 1e-12) {
            return yKnots[lo];
        }
        double t = (x - xKnots[lo]) / span;
        return yKnots[lo] + t * (yKnots[lo + 1] - yKnots[lo]);
    }

    /**
     * Evaluate the function at multiple points.
     *
     * @param xs input values
     * @return interpolated values, same length as xs
     */
    public double[] evaluate(double[] xs) {
        double[] result = new double[xs.length];
        for (int i = 0; i < xs.length; i++) {
            result[i] = evaluate(xs[i]);
        }
        return result;
    }

    /** Number of knots. */
    public int numKnots() {
        return xKnots.length;
    }

    /** Copy of x-knot positions. */
    public double[] getXKnots() {
        return xKnots.clone();
    }

    /** Copy of y-knot values. */
    public double[] getYKnots() {
        return yKnots.clone();
    }
}
