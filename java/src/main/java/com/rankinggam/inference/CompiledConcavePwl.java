package com.rankinggam.inference;

/**
 * Compiled concave piecewise-linear evaluator — no loops, all constants inlined.
 *
 * <p>Mirrors the {@link CompiledPwlFunction} pattern for concave diversity towers.
 * At construction time, selects a K-specialized lambda that bakes all segment
 * parameters (edges, widths, slopes) as {@code final} locals:
 * <ul>
 *   <li>K=1..6: flat if/else chain, all values in registers/JIT constants
 *   <li>K>6: fallback to cloned-array loop
 * </ul>
 *
 * <p>Concave PWL formula:
 * {@code result = intercept + Σ slopes[i] * clamp(x - edges[i], 0, widths[i])}
 *
 * <p>Usage:
 * <pre>
 *   CompiledConcavePwl compiled = CompiledConcavePwl.compile(concavePwl);
 *   double y = compiled.evaluate(0.7);
 *   double yMax = compiled.evaluateAtMax(); // precomputed constant
 * </pre>
 */
public final class CompiledConcavePwl {

    @FunctionalInterface
    public interface ConcaveEval {
        double eval(double x);
    }

    private final ConcaveEval evaluator;
    private final double atMax; // precomputed evaluate(xMax)

    /**
     * Compile a ConcavePwlFunction into a specialized evaluator.
     */
    public static CompiledConcavePwl compile(ConcavePwlFunction src) {
        return new CompiledConcavePwl(src);
    }

    private CompiledConcavePwl(ConcavePwlFunction src) {
        // Delegate to the raw-parameter constructor for full compilation
        this(src.intercept(), src.knotEdges(), src.knotWidths(), src.slopes(),
             src.xMin(), src.xMax());
    }

    /**
     * Compile from raw concave PWL parameters.
     *
     * @param intercept  bias term
     * @param knotEdges  left edges of each segment
     * @param knotWidths widths of each segment
     * @param slopes     slopes per segment (monotone non-increasing, non-negative)
     * @param xMin       clamp minimum
     * @param xMax       clamp maximum
     */
    public static CompiledConcavePwl compile(double intercept, double[] knotEdges,
                                              double[] knotWidths, double[] slopes,
                                              double xMin, double xMax) {
        return new CompiledConcavePwl(intercept, knotEdges, knotWidths, slopes, xMin, xMax);
    }

    private CompiledConcavePwl(double intercept, double[] edges, double[] widths,
                                double[] slopes, double xMin, double xMax) {
        int k = edges.length;

        switch (k) {
            case 0:
                this.evaluator = compileK0(intercept, xMin, xMax);
                break;
            case 1:
                this.evaluator = compileK1(intercept, edges, widths, slopes, xMin, xMax);
                break;
            case 2:
                this.evaluator = compileK2(intercept, edges, widths, slopes, xMin, xMax);
                break;
            case 3:
                this.evaluator = compileK3(intercept, edges, widths, slopes, xMin, xMax);
                break;
            case 4:
                this.evaluator = compileK4(intercept, edges, widths, slopes, xMin, xMax);
                break;
            case 5:
                this.evaluator = compileK5(intercept, edges, widths, slopes, xMin, xMax);
                break;
            case 6:
                this.evaluator = compileK6(intercept, edges, widths, slopes, xMin, xMax);
                break;
            default:
                this.evaluator = compileFallback(intercept, edges, widths, slopes, xMin, xMax, k);
                break;
        }

        // Precompute atMax
        this.atMax = this.evaluator.eval(xMax);
    }

    /** Evaluate at a single point. */
    public double evaluate(double x) {
        return evaluator.eval(x);
    }

    /** Precomputed value at xMax (max diversity). */
    public double evaluateAtMax() {
        return atMax;
    }

    /** The raw compiled evaluator lambda — for direct inline use. */
    public ConcaveEval evaluator() {
        return evaluator;
    }

    // ── K=0: constant ──

    private static ConcaveEval compileK0(double intercept, double xMin, double xMax) {
        final double r = intercept;
        return x -> r;
    }

    // ── K=1: single segment ──

    private static ConcaveEval compileK1(double intercept, double[] e, double[] w,
                                          double[] s, double xMin, double xMax) {
        final double i0 = intercept, lo = xMin, hi = xMax;
        final double e0 = e[0], w0 = w[0], s0 = s[0];
        return x -> {
            if (x < lo) x = lo; else if (x > hi) x = hi;
            double seg = x - e0;
            if (seg <= 0) return i0;
            return i0 + s0 * (seg > w0 ? w0 : seg);
        };
    }

    // ── K=2 ──

    private static ConcaveEval compileK2(double intercept, double[] e, double[] w,
                                          double[] s, double xMin, double xMax) {
        final double i0 = intercept, lo = xMin, hi = xMax;
        final double e0 = e[0], e1 = e[1];
        final double w0 = w[0], w1 = w[1];
        final double s0 = s[0], s1 = s[1];
        return x -> {
            if (x < lo) x = lo; else if (x > hi) x = hi;
            double r = i0;
            double seg = x - e0; if (seg <= 0) return r;
            r += s0 * (seg > w0 ? w0 : seg);
            seg = x - e1; if (seg <= 0) return r;
            r += s1 * (seg > w1 ? w1 : seg);
            return r;
        };
    }

    // ── K=3 ──

    private static ConcaveEval compileK3(double intercept, double[] e, double[] w,
                                          double[] s, double xMin, double xMax) {
        final double i0 = intercept, lo = xMin, hi = xMax;
        final double e0 = e[0], e1 = e[1], e2 = e[2];
        final double w0 = w[0], w1 = w[1], w2 = w[2];
        final double s0 = s[0], s1 = s[1], s2 = s[2];
        return x -> {
            if (x < lo) x = lo; else if (x > hi) x = hi;
            double r = i0;
            double seg = x - e0; if (seg <= 0) return r;
            r += s0 * (seg > w0 ? w0 : seg);
            seg = x - e1; if (seg <= 0) return r;
            r += s1 * (seg > w1 ? w1 : seg);
            seg = x - e2; if (seg <= 0) return r;
            r += s2 * (seg > w2 ? w2 : seg);
            return r;
        };
    }

    // ── K=4 (benchmark default) ──

    private static ConcaveEval compileK4(double intercept, double[] e, double[] w,
                                          double[] s, double xMin, double xMax) {
        final double i0 = intercept, lo = xMin, hi = xMax;
        final double e0 = e[0], e1 = e[1], e2 = e[2], e3 = e[3];
        final double w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
        final double s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3];
        return x -> {
            if (x < lo) x = lo; else if (x > hi) x = hi;
            double r = i0;
            double seg = x - e0; if (seg <= 0) return r;
            r += s0 * (seg > w0 ? w0 : seg);
            seg = x - e1; if (seg <= 0) return r;
            r += s1 * (seg > w1 ? w1 : seg);
            seg = x - e2; if (seg <= 0) return r;
            r += s2 * (seg > w2 ? w2 : seg);
            seg = x - e3; if (seg <= 0) return r;
            r += s3 * (seg > w3 ? w3 : seg);
            return r;
        };
    }

    // ── K=5 ──

    private static ConcaveEval compileK5(double intercept, double[] e, double[] w,
                                          double[] s, double xMin, double xMax) {
        final double i0 = intercept, lo = xMin, hi = xMax;
        final double e0 = e[0], e1 = e[1], e2 = e[2], e3 = e[3], e4 = e[4];
        final double w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4];
        final double s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3], s4 = s[4];
        return x -> {
            if (x < lo) x = lo; else if (x > hi) x = hi;
            double r = i0;
            double seg = x - e0; if (seg <= 0) return r;
            r += s0 * (seg > w0 ? w0 : seg);
            seg = x - e1; if (seg <= 0) return r;
            r += s1 * (seg > w1 ? w1 : seg);
            seg = x - e2; if (seg <= 0) return r;
            r += s2 * (seg > w2 ? w2 : seg);
            seg = x - e3; if (seg <= 0) return r;
            r += s3 * (seg > w3 ? w3 : seg);
            seg = x - e4; if (seg <= 0) return r;
            r += s4 * (seg > w4 ? w4 : seg);
            return r;
        };
    }

    // ── K=6 ──

    private static ConcaveEval compileK6(double intercept, double[] e, double[] w,
                                          double[] s, double xMin, double xMax) {
        final double i0 = intercept, lo = xMin, hi = xMax;
        final double e0 = e[0], e1 = e[1], e2 = e[2], e3 = e[3], e4 = e[4], e5 = e[5];
        final double w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5];
        final double s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3], s4 = s[4], s5 = s[5];
        return x -> {
            if (x < lo) x = lo; else if (x > hi) x = hi;
            double r = i0;
            double seg = x - e0; if (seg <= 0) return r;
            r += s0 * (seg > w0 ? w0 : seg);
            seg = x - e1; if (seg <= 0) return r;
            r += s1 * (seg > w1 ? w1 : seg);
            seg = x - e2; if (seg <= 0) return r;
            r += s2 * (seg > w2 ? w2 : seg);
            seg = x - e3; if (seg <= 0) return r;
            r += s3 * (seg > w3 ? w3 : seg);
            seg = x - e4; if (seg <= 0) return r;
            r += s4 * (seg > w4 ? w4 : seg);
            seg = x - e5; if (seg <= 0) return r;
            r += s5 * (seg > w5 ? w5 : seg);
            return r;
        };
    }

    // ── Fallback for K>6 ──

    private static ConcaveEval compileFallback(double intercept, double[] edges,
                                                double[] widths, double[] slopes,
                                                double xMin, double xMax, int k) {
        final double i0 = intercept, lo = xMin, hi = xMax;
        final double[] e = edges.clone(), w = widths.clone(), s = slopes.clone();
        final int len = k;
        return x -> {
            if (x < lo) x = lo; else if (x > hi) x = hi;
            double r = i0;
            for (int j = 0; j < len; j++) {
                double seg = x - e[j];
                if (seg <= 0) break;
                if (seg > w[j]) seg = w[j];
                r += s[j] * seg;
            }
            return r;
        };
    }
}
