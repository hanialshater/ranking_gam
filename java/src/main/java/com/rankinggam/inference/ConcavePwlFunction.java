package com.rankinggam.inference;

/**
 * Concave piecewise-linear function for diversity towers.
 *
 * <p>Evaluates: {@code result = intercept + sum_i(slopes[i] * clamp(x - edges[i], 0, widths[i]))}
 *
 * <p>Slopes satisfy {@code s[0] >= s[1] >= ... >= s[K-1] >= 0} (monotone + concave).
 * This guarantees submodularity when composed with modular groupwise features,
 * so greedy reranking achieves a (1-1/e) approximation guarantee.
 *
 * <p>These towers are already piecewise-linear by construction (no distillation
 * needed). Parameters are exported directly from Python's {@code ConcavePWL} module.
 */
public final class ConcavePwlFunction {

    private final double intercept;
    private final double[] knotEdges;   // left edge of each segment (K values)
    private final double[] knotWidths;  // width of each segment (K values)
    private final double[] slopes;      // slopes[i] >= slopes[i+1] >= 0
    private final double xMin;
    private final double xMax;
    private final int numKnots;

    /**
     * @param intercept  bias term
     * @param knotEdges  left edges of each segment (length K)
     * @param knotWidths widths of each segment (length K)
     * @param slopes     slopes per segment, monotone non-increasing and non-negative
     * @param xMin       minimum input value (clamp below)
     * @param xMax       maximum input value (clamp above)
     */
    public ConcavePwlFunction(double intercept, double[] knotEdges,
                               double[] knotWidths, double[] slopes,
                               double xMin, double xMax) {
        if (knotEdges.length != knotWidths.length || knotEdges.length != slopes.length) {
            throw new IllegalArgumentException("knotEdges, knotWidths, slopes must have same length");
        }
        this.intercept = intercept;
        this.knotEdges = knotEdges.clone();
        this.knotWidths = knotWidths.clone();
        this.slopes = slopes.clone();
        this.xMin = xMin;
        this.xMax = xMax;
        this.numKnots = slopes.length;
    }

    /**
     * Evaluate the concave PWL function at a single point.
     */
    public double evaluate(double x) {
        if (x < xMin) x = xMin;
        else if (x > xMax) x = xMax;

        double result = intercept;
        for (int i = 0; i < numKnots; i++) {
            double seg = x - knotEdges[i];
            if (seg <= 0) break; // knot_edges sorted: all subsequent segments are 0
            if (seg > knotWidths[i]) seg = knotWidths[i];
            result += slopes[i] * seg;
        }
        return result;
    }

    /**
     * Evaluate at the maximum input (x_max). Useful for initial upper-bound
     * estimates in lazy greedy (diversity is maximized at x_max for concave towers).
     */
    public double evaluateAtMax() {
        return evaluate(xMax);
    }

    public double intercept() { return intercept; }
    public double xMin() { return xMin; }
    public double xMax() { return xMax; }
    public int numKnots() { return numKnots; }
    public double[] knotEdges() { return knotEdges.clone(); }
    public double[] knotWidths() { return knotWidths.clone(); }
    public double[] slopes() { return slopes.clone(); }
}
