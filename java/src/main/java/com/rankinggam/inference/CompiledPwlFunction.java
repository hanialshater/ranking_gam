package com.rankinggam.inference;

/**
 * Compiled piecewise-linear function that bakes knot values into
 * specialized if/else evaluators — no loops, no binary search.
 *
 * <p>At construction time, selects a specialized implementation based on
 * the number of knots (K). For typical GAM distillation (K=2..6), the
 * evaluator is a flat if/else chain with all knot positions, y-values,
 * and slopes inlined as local finals. This eliminates:
 * <ul>
 *   <li>Array access overhead (values are in registers / JIT constants)
 *   <li>Loop control flow (no iteration, no branch misprediction from loop)
 *   <li>Binary search overhead (no comparisons beyond the minimum)
 * </ul>
 *
 * <p>For K>6 knots, falls back to precomputed-slope binary search.
 *
 * <p>The bulk {@link #evaluateAndAccumulate} method applies the same
 * compiled evaluator across a column of feature values (all documents
 * for one feature), accumulating into a scores array.
 */
public final class CompiledPwlFunction {

    @FunctionalInterface
    interface PwlEvaluator {
        double eval(double x);
    }

    @FunctionalInterface
    interface PwlBulkEvaluator {
        void evalBulk(double[] values, double[] scores, int count);
    }

    private final PwlEvaluator evaluator;
    private final PwlBulkEvaluator bulkEvaluator;
    private final int n;
    private final double[] xKnots;
    private final double[] yKnots;

    /**
     * Compile a PwlFunction into an optimized evaluator.
     */
    public static CompiledPwlFunction compile(PwlFunction pwl) {
        return new CompiledPwlFunction(pwl.xKnots, pwl.yKnots, pwl.slopes, pwl.n);
    }

    /**
     * Compile from raw knot arrays.
     */
    public static CompiledPwlFunction compile(double[] xKnots, double[] yKnots) {
        PwlFunction pwl = new PwlFunction(xKnots, yKnots);
        return compile(pwl);
    }

    private CompiledPwlFunction(double[] xKnots, double[] yKnots, double[] slopes, int n) {
        this.n = n;
        this.xKnots = xKnots.clone();
        this.yKnots = yKnots.clone();

        switch (n) {
            case 1:
                this.evaluator = compileK1(yKnots[0]);
                this.bulkEvaluator = compileBulkK1(yKnots[0]);
                break;
            case 2:
                this.evaluator = compileK2(xKnots, yKnots, slopes);
                this.bulkEvaluator = compileBulkK2(xKnots, yKnots, slopes);
                break;
            case 3:
                this.evaluator = compileK3(xKnots, yKnots, slopes);
                this.bulkEvaluator = compileBulkK3(xKnots, yKnots, slopes);
                break;
            case 4:
                this.evaluator = compileK4(xKnots, yKnots, slopes);
                this.bulkEvaluator = compileBulkK4(xKnots, yKnots, slopes);
                break;
            case 5:
                this.evaluator = compileK5(xKnots, yKnots, slopes);
                this.bulkEvaluator = compileBulkK5(xKnots, yKnots, slopes);
                break;
            case 6:
                this.evaluator = compileK6(xKnots, yKnots, slopes);
                this.bulkEvaluator = compileBulkK6(xKnots, yKnots, slopes);
                break;
            default:
                this.evaluator = compileFallback(xKnots, yKnots, slopes, n);
                this.bulkEvaluator = compileBulkFallback(xKnots, yKnots, slopes, n);
                break;
        }
    }

    /** Evaluate at a single point. */
    public double evaluate(double x) {
        return evaluator.eval(x);
    }

    /** Bulk evaluate: add PWL contributions into scores accumulator. */
    public void evaluateAndAccumulate(double[] values, double[] scores, int count) {
        bulkEvaluator.evalBulk(values, scores, count);
    }

    // ── K=1: constant function ──

    private static PwlEvaluator compileK1(double y) {
        return x -> y;
    }

    private static PwlBulkEvaluator compileBulkK1(double y) {
        return (values, scores, count) -> {
            for (int i = 0; i < count; i++) scores[i] += y;
        };
    }

    // ── K=2: single segment ──

    private static PwlEvaluator compileK2(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1];
        final double y0 = yk[0], y1 = yk[1];
        final double s0 = sl[0];
        return x -> {
            if (x <= x0) return y0;
            if (x >= x1) return y1;
            return y0 + s0 * (x - x0);
        };
    }

    private static PwlBulkEvaluator compileBulkK2(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1];
        final double y0 = yk[0], y1 = yk[1];
        final double s0 = sl[0];
        return (values, scores, count) -> {
            for (int i = 0; i < count; i++) {
                double x = values[i];
                if (x <= x0) { scores[i] += y0; }
                else if (x >= x1) { scores[i] += y1; }
                else { scores[i] += y0 + s0 * (x - x0); }
            }
        };
    }

    // ── K=3: two segments ──

    private static PwlEvaluator compileK3(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1], x2 = xk[2];
        final double y0 = yk[0], y1 = yk[1], y2 = yk[2];
        final double s0 = sl[0], s1 = sl[1];
        return x -> {
            if (x <= x0) return y0;
            if (x >= x2) return y2;
            if (x <= x1) return y0 + s0 * (x - x0);
            return y1 + s1 * (x - x1);
        };
    }

    private static PwlBulkEvaluator compileBulkK3(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1], x2 = xk[2];
        final double y0 = yk[0], y1 = yk[1], y2 = yk[2];
        final double s0 = sl[0], s1 = sl[1];
        return (values, scores, count) -> {
            for (int i = 0; i < count; i++) {
                double x = values[i];
                if (x <= x0) { scores[i] += y0; }
                else if (x >= x2) { scores[i] += y2; }
                else if (x <= x1) { scores[i] += y0 + s0 * (x - x0); }
                else { scores[i] += y1 + s1 * (x - x1); }
            }
        };
    }

    // ── K=4: three segments ──

    private static PwlEvaluator compileK4(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1], x2 = xk[2], x3 = xk[3];
        final double y0 = yk[0], y1 = yk[1], y2 = yk[2], y3 = yk[3];
        final double s0 = sl[0], s1 = sl[1], s2 = sl[2];
        return x -> {
            if (x <= x0) return y0;
            if (x >= x3) return y3;
            if (x <= x1) return y0 + s0 * (x - x0);
            if (x <= x2) return y1 + s1 * (x - x1);
            return y2 + s2 * (x - x2);
        };
    }

    private static PwlBulkEvaluator compileBulkK4(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1], x2 = xk[2], x3 = xk[3];
        final double y0 = yk[0], y1 = yk[1], y2 = yk[2], y3 = yk[3];
        final double s0 = sl[0], s1 = sl[1], s2 = sl[2];
        return (values, scores, count) -> {
            for (int i = 0; i < count; i++) {
                double x = values[i];
                if (x <= x0) { scores[i] += y0; }
                else if (x >= x3) { scores[i] += y3; }
                else if (x <= x1) { scores[i] += y0 + s0 * (x - x0); }
                else if (x <= x2) { scores[i] += y1 + s1 * (x - x1); }
                else { scores[i] += y2 + s2 * (x - x2); }
            }
        };
    }

    // ── K=5: four segments ──

    private static PwlEvaluator compileK5(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1], x2 = xk[2], x3 = xk[3], x4 = xk[4];
        final double y0 = yk[0], y1 = yk[1], y2 = yk[2], y3 = yk[3], y4 = yk[4];
        final double s0 = sl[0], s1 = sl[1], s2 = sl[2], s3 = sl[3];
        return x -> {
            if (x <= x0) return y0;
            if (x >= x4) return y4;
            if (x <= x2) {
                if (x <= x1) return y0 + s0 * (x - x0);
                return y1 + s1 * (x - x1);
            }
            if (x <= x3) return y2 + s2 * (x - x2);
            return y3 + s3 * (x - x3);
        };
    }

    private static PwlBulkEvaluator compileBulkK5(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1], x2 = xk[2], x3 = xk[3], x4 = xk[4];
        final double y0 = yk[0], y1 = yk[1], y2 = yk[2], y3 = yk[3], y4 = yk[4];
        final double s0 = sl[0], s1 = sl[1], s2 = sl[2], s3 = sl[3];
        return (values, scores, count) -> {
            for (int i = 0; i < count; i++) {
                double x = values[i];
                if (x <= x0) { scores[i] += y0; }
                else if (x >= x4) { scores[i] += y4; }
                else if (x <= x2) {
                    if (x <= x1) { scores[i] += y0 + s0 * (x - x0); }
                    else { scores[i] += y1 + s1 * (x - x1); }
                }
                else if (x <= x3) { scores[i] += y2 + s2 * (x - x2); }
                else { scores[i] += y3 + s3 * (x - x3); }
            }
        };
    }

    // ── K=6: five segments ──

    private static PwlEvaluator compileK6(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1], x2 = xk[2], x3 = xk[3], x4 = xk[4], x5 = xk[5];
        final double y0 = yk[0], y1 = yk[1], y2 = yk[2], y3 = yk[3], y4 = yk[4], y5 = yk[5];
        final double s0 = sl[0], s1 = sl[1], s2 = sl[2], s3 = sl[3], s4 = sl[4];
        return x -> {
            if (x <= x0) return y0;
            if (x >= x5) return y5;
            // Binary-style: split at midpoint x2
            if (x <= x2) {
                if (x <= x1) return y0 + s0 * (x - x0);
                return y1 + s1 * (x - x1);
            }
            if (x <= x3) return y2 + s2 * (x - x2);
            if (x <= x4) return y3 + s3 * (x - x3);
            return y4 + s4 * (x - x4);
        };
    }

    private static PwlBulkEvaluator compileBulkK6(double[] xk, double[] yk, double[] sl) {
        final double x0 = xk[0], x1 = xk[1], x2 = xk[2], x3 = xk[3], x4 = xk[4], x5 = xk[5];
        final double y0 = yk[0], y1 = yk[1], y2 = yk[2], y3 = yk[3], y4 = yk[4], y5 = yk[5];
        final double s0 = sl[0], s1 = sl[1], s2 = sl[2], s3 = sl[3], s4 = sl[4];
        return (values, scores, count) -> {
            for (int i = 0; i < count; i++) {
                double x = values[i];
                if (x <= x0) { scores[i] += y0; }
                else if (x >= x5) { scores[i] += y5; }
                else if (x <= x2) {
                    if (x <= x1) { scores[i] += y0 + s0 * (x - x0); }
                    else { scores[i] += y1 + s1 * (x - x1); }
                }
                else if (x <= x3) { scores[i] += y2 + s2 * (x - x2); }
                else if (x <= x4) { scores[i] += y3 + s3 * (x - x3); }
                else { scores[i] += y4 + s4 * (x - x4); }
            }
        };
    }

    // ── Fallback for K>6: precomputed slopes + binary search ──

    private static PwlEvaluator compileFallback(double[] xk, double[] yk, double[] sl, int n) {
        final double[] x = xk.clone(), y = yk.clone(), s = sl.clone();
        final int len = n;
        return xv -> {
            if (xv <= x[0]) return y[0];
            if (xv >= x[len - 1]) return y[len - 1];
            int lo = 0, hi = len - 1;
            while (lo < hi - 1) {
                int mid = (lo + hi) >>> 1;
                if (x[mid] <= xv) lo = mid; else hi = mid;
            }
            return y[lo] + s[lo] * (xv - x[lo]);
        };
    }

    private static PwlBulkEvaluator compileBulkFallback(double[] xk, double[] yk, double[] sl, int n) {
        final double[] x = xk.clone(), y = yk.clone(), s = sl.clone();
        final int len = n;
        final double x0 = x[0], xEnd = x[len - 1], y0 = y[0], yEnd = y[len - 1];
        return (values, scores, count) -> {
            for (int i = 0; i < count; i++) {
                double xv = values[i];
                if (xv <= x0) { scores[i] += y0; continue; }
                if (xv >= xEnd) { scores[i] += yEnd; continue; }
                int lo = 0, hi = len - 1;
                while (lo < hi - 1) {
                    int mid = (lo + hi) >>> 1;
                    if (x[mid] <= xv) lo = mid; else hi = mid;
                }
                scores[i] += y[lo] + s[lo] * (xv - x[lo]);
            }
        };
    }

    public int numKnots() { return n; }
    public double[] getXKnots() { return xKnots.clone(); }
    public double[] getYKnots() { return yKnots.clone(); }
}
