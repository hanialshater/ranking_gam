package com.rankinggam.inference;

/**
 * Compiled bilinear grid evaluator that bakes grid boundaries into
 * specialized if/else cell finders — no binary search at eval time.
 *
 * <p>At construction time, selects a specialized implementation based on
 * grid dimensions (N×N for N=2..6). For typical GA2M interactions (5×5 grids),
 * the evaluator is a flat if/else chain with all grid boundaries, inverse
 * segment widths, and z-values inlined as local finals. This eliminates:
 * <ul>
 *   <li>Binary search overhead on both axes (replaced by if/else tree)
 *   <li>Array access overhead for grid boundaries (values in registers/JIT constants)
 *   <li>Division in interpolation weights (precomputed inverse widths)
 * </ul>
 *
 * <p>For grids larger than 6×6 or non-square grids, falls back to
 * precomputed-inverse binary search (still faster than original).
 */
public final class CompiledBilinearGridFunction {

    @FunctionalInterface
    interface GridEvaluator {
        double eval(double x1, double x2);
    }

    @FunctionalInterface
    interface GridBulkEvaluator {
        void evalBulk(double[] x1Values, double[] x2Values, double[] scores, int count);
    }

    private final GridEvaluator evaluator;
    private final GridBulkEvaluator bulkEvaluator;
    private final int n1, n2;

    /**
     * Compile a BilinearGridFunction into an optimized evaluator.
     */
    public static CompiledBilinearGridFunction compile(BilinearGridFunction grid) {
        return new CompiledBilinearGridFunction(grid.x1Grid, grid.x2Grid, grid.z);
    }

    /**
     * Compile from raw grid arrays.
     */
    public static CompiledBilinearGridFunction compile(double[] x1Grid, double[] x2Grid, double[][] z) {
        return new CompiledBilinearGridFunction(x1Grid, x2Grid, z);
    }

    private CompiledBilinearGridFunction(double[] x1Grid, double[] x2Grid, double[][] z) {
        this.n1 = x1Grid.length;
        this.n2 = x2Grid.length;

        // Precompute inverse segment widths and flatten z
        double[] invDx1 = new double[n1 - 1];
        for (int i = 0; i < n1 - 1; i++) {
            double dx = x1Grid[i + 1] - x1Grid[i];
            invDx1[i] = dx < 1e-12 ? 0.0 : 1.0 / dx;
        }
        double[] invDx2 = new double[n2 - 1];
        for (int j = 0; j < n2 - 1; j++) {
            double dx = x2Grid[j + 1] - x2Grid[j];
            invDx2[j] = dx < 1e-12 ? 0.0 : 1.0 / dx;
        }
        double[] flatZ = new double[n1 * n2];
        for (int i = 0; i < n1; i++) {
            System.arraycopy(z[i], 0, flatZ, i * n2, n2);
        }

        // Select specialized implementation for square grids
        if (n1 == n2) {
            switch (n1) {
                case 2:
                    this.evaluator = compileSquare2(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    this.bulkEvaluator = compileBulkSquare2(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    return;
                case 3:
                    this.evaluator = compileSquare3(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    this.bulkEvaluator = compileBulkSquare3(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    return;
                case 4:
                    this.evaluator = compileSquare4(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    this.bulkEvaluator = compileBulkSquare4(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    return;
                case 5:
                    this.evaluator = compileSquare5(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    this.bulkEvaluator = compileBulkSquare5(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    return;
                case 6:
                    this.evaluator = compileSquare6(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    this.bulkEvaluator = compileBulkSquare6(x1Grid, x2Grid, invDx1, invDx2, flatZ);
                    return;
            }
        }

        // Fallback for non-square or large grids
        this.evaluator = compileFallback(x1Grid, x2Grid, invDx1, invDx2, flatZ, n1, n2);
        this.bulkEvaluator = compileBulkFallback(x1Grid, x2Grid, invDx1, invDx2, flatZ, n1, n2);
    }

    /** Evaluate at a single (x1, x2) point. */
    public double evaluate(double x1, double x2) {
        return evaluator.eval(x1, x2);
    }

    /** Bulk evaluate: add bilinear contributions into scores accumulator. */
    public void evaluateAndAccumulate(double[] x1Values, double[] x2Values,
                                       double[] scores, int count) {
        bulkEvaluator.evalBulk(x1Values, x2Values, scores, count);
    }

    // ── Bilinear interpolation helper ──

    private static double bilinearInterp(double s, double t,
                                          double z00, double z10, double z01, double z11) {
        return (1 - s) * ((1 - t) * z00 + t * z01)
                + s * ((1 - t) * z10 + t * z11);
    }

    // ── 2×2: single cell, no search ──

    private static GridEvaluator compileSquare2(double[] x1g, double[] x2g,
                                                 double[] id1, double[] id2, double[] fz) {
        final double x1Lo = x1g[0], x1Hi = x1g[1];
        final double x2Lo = x2g[0], x2Hi = x2g[1];
        final double inv1 = id1[0], inv2 = id2[0];
        final double z00 = fz[0], z01 = fz[1], z10 = fz[2], z11 = fz[3];
        return (x1, x2) -> {
            if (x1 < x1Lo) x1 = x1Lo; else if (x1 > x1Hi) x1 = x1Hi;
            if (x2 < x2Lo) x2 = x2Lo; else if (x2 > x2Hi) x2 = x2Hi;
            double s = (x1 - x1Lo) * inv1;
            double t = (x2 - x2Lo) * inv2;
            return bilinearInterp(s, t, z00, z10, z01, z11);
        };
    }

    private static GridBulkEvaluator compileBulkSquare2(double[] x1g, double[] x2g,
                                                         double[] id1, double[] id2, double[] fz) {
        final double x1Lo = x1g[0], x1Hi = x1g[1];
        final double x2Lo = x2g[0], x2Hi = x2g[1];
        final double inv1 = id1[0], inv2 = id2[0];
        final double z00 = fz[0], z01 = fz[1], z10 = fz[2], z11 = fz[3];
        return (x1Values, x2Values, scores, count) -> {
            for (int idx = 0; idx < count; idx++) {
                double x1 = x1Values[idx];
                double x2 = x2Values[idx];
                if (x1 < x1Lo) x1 = x1Lo; else if (x1 > x1Hi) x1 = x1Hi;
                if (x2 < x2Lo) x2 = x2Lo; else if (x2 > x2Hi) x2 = x2Hi;
                double s = (x1 - x1Lo) * inv1;
                double t = (x2 - x2Lo) * inv2;
                scores[idx] += bilinearInterp(s, t, z00, z10, z01, z11);
            }
        };
    }

    // ── 3×3: 2×2 cells, single comparison per axis ──

    private static GridEvaluator compileSquare3(double[] x1g, double[] x2g,
                                                 double[] id1, double[] id2, double[] fz) {
        final double x1_0 = x1g[0], x1_1 = x1g[1], x1_2 = x1g[2];
        final double x2_0 = x2g[0], x2_1 = x2g[1], x2_2 = x2g[2];
        final double id1_0 = id1[0], id1_1 = id1[1];
        final double id2_0 = id2[0], id2_1 = id2[1];
        final double[] z = fz.clone();
        final int n = 3;
        return (x1, x2) -> {
            if (x1 < x1_0) x1 = x1_0; else if (x1 > x1_2) x1 = x1_2;
            if (x2 < x2_0) x2 = x2_0; else if (x2 > x2_2) x2 = x2_2;
            int i; double seg1, inv1;
            if (x1 <= x1_1) { i = 0; seg1 = x1_0; inv1 = id1_0; }
            else             { i = 1; seg1 = x1_1; inv1 = id1_1; }
            int j; double seg2, inv2;
            if (x2 <= x2_1) { j = 0; seg2 = x2_0; inv2 = id2_0; }
            else             { j = 1; seg2 = x2_1; inv2 = id2_1; }
            double s = (x1 - seg1) * inv1;
            double t = (x2 - seg2) * inv2;
            int b = i * n + j;
            return bilinearInterp(s, t, z[b], z[b + n], z[b + 1], z[b + n + 1]);
        };
    }

    private static GridBulkEvaluator compileBulkSquare3(double[] x1g, double[] x2g,
                                                         double[] id1, double[] id2, double[] fz) {
        final double x1_0 = x1g[0], x1_1 = x1g[1], x1_2 = x1g[2];
        final double x2_0 = x2g[0], x2_1 = x2g[1], x2_2 = x2g[2];
        final double id1_0 = id1[0], id1_1 = id1[1];
        final double id2_0 = id2[0], id2_1 = id2[1];
        final double[] z = fz.clone();
        final int n = 3;
        return (x1Values, x2Values, scores, count) -> {
            for (int idx = 0; idx < count; idx++) {
                double x1 = x1Values[idx];
                double x2 = x2Values[idx];
                if (x1 < x1_0) x1 = x1_0; else if (x1 > x1_2) x1 = x1_2;
                if (x2 < x2_0) x2 = x2_0; else if (x2 > x2_2) x2 = x2_2;
                int i; double seg1, inv1;
                if (x1 <= x1_1) { i = 0; seg1 = x1_0; inv1 = id1_0; }
                else             { i = 1; seg1 = x1_1; inv1 = id1_1; }
                int j; double seg2, inv2;
                if (x2 <= x2_1) { j = 0; seg2 = x2_0; inv2 = id2_0; }
                else             { j = 1; seg2 = x2_1; inv2 = id2_1; }
                double s = (x1 - seg1) * inv1;
                double t = (x2 - seg2) * inv2;
                int b = i * n + j;
                scores[idx] += bilinearInterp(s, t, z[b], z[b + n], z[b + 1], z[b + n + 1]);
            }
        };
    }

    // ── 4×4: 3×3 cells, two comparisons per axis ──

    private static GridEvaluator compileSquare4(double[] x1g, double[] x2g,
                                                 double[] id1, double[] id2, double[] fz) {
        final double x1_0 = x1g[0], x1_1 = x1g[1], x1_2 = x1g[2], x1_3 = x1g[3];
        final double x2_0 = x2g[0], x2_1 = x2g[1], x2_2 = x2g[2], x2_3 = x2g[3];
        final double id1_0 = id1[0], id1_1 = id1[1], id1_2 = id1[2];
        final double id2_0 = id2[0], id2_1 = id2[1], id2_2 = id2[2];
        final double[] z = fz.clone();
        final int n = 4;
        return (x1, x2) -> {
            if (x1 < x1_0) x1 = x1_0; else if (x1 > x1_3) x1 = x1_3;
            if (x2 < x2_0) x2 = x2_0; else if (x2 > x2_3) x2 = x2_3;
            int i; double seg1, inv1;
            if (x1 <= x1_1) { i = 0; seg1 = x1_0; inv1 = id1_0; }
            else if (x1 <= x1_2) { i = 1; seg1 = x1_1; inv1 = id1_1; }
            else { i = 2; seg1 = x1_2; inv1 = id1_2; }
            int j; double seg2, inv2;
            if (x2 <= x2_1) { j = 0; seg2 = x2_0; inv2 = id2_0; }
            else if (x2 <= x2_2) { j = 1; seg2 = x2_1; inv2 = id2_1; }
            else { j = 2; seg2 = x2_2; inv2 = id2_2; }
            double s = (x1 - seg1) * inv1;
            double t = (x2 - seg2) * inv2;
            int b = i * n + j;
            return bilinearInterp(s, t, z[b], z[b + n], z[b + 1], z[b + n + 1]);
        };
    }

    private static GridBulkEvaluator compileBulkSquare4(double[] x1g, double[] x2g,
                                                         double[] id1, double[] id2, double[] fz) {
        final double x1_0 = x1g[0], x1_1 = x1g[1], x1_2 = x1g[2], x1_3 = x1g[3];
        final double x2_0 = x2g[0], x2_1 = x2g[1], x2_2 = x2g[2], x2_3 = x2g[3];
        final double id1_0 = id1[0], id1_1 = id1[1], id1_2 = id1[2];
        final double id2_0 = id2[0], id2_1 = id2[1], id2_2 = id2[2];
        final double[] z = fz.clone();
        final int n = 4;
        return (x1Values, x2Values, scores, count) -> {
            for (int idx = 0; idx < count; idx++) {
                double x1 = x1Values[idx];
                double x2 = x2Values[idx];
                if (x1 < x1_0) x1 = x1_0; else if (x1 > x1_3) x1 = x1_3;
                if (x2 < x2_0) x2 = x2_0; else if (x2 > x2_3) x2 = x2_3;
                int i; double seg1, inv1;
                if (x1 <= x1_1) { i = 0; seg1 = x1_0; inv1 = id1_0; }
                else if (x1 <= x1_2) { i = 1; seg1 = x1_1; inv1 = id1_1; }
                else { i = 2; seg1 = x1_2; inv1 = id1_2; }
                int j; double seg2, inv2;
                if (x2 <= x2_1) { j = 0; seg2 = x2_0; inv2 = id2_0; }
                else if (x2 <= x2_2) { j = 1; seg2 = x2_1; inv2 = id2_1; }
                else { j = 2; seg2 = x2_2; inv2 = id2_2; }
                double s = (x1 - seg1) * inv1;
                double t = (x2 - seg2) * inv2;
                int b = i * n + j;
                scores[idx] += bilinearInterp(s, t, z[b], z[b + n], z[b + 1], z[b + n + 1]);
            }
        };
    }

    // ── 5×5: 4×4 cells, binary-split if/else per axis ──

    private static GridEvaluator compileSquare5(double[] x1g, double[] x2g,
                                                 double[] id1, double[] id2, double[] fz) {
        final double x1_0 = x1g[0], x1_1 = x1g[1], x1_2 = x1g[2], x1_3 = x1g[3], x1_4 = x1g[4];
        final double x2_0 = x2g[0], x2_1 = x2g[1], x2_2 = x2g[2], x2_3 = x2g[3], x2_4 = x2g[4];
        final double id1_0 = id1[0], id1_1 = id1[1], id1_2 = id1[2], id1_3 = id1[3];
        final double id2_0 = id2[0], id2_1 = id2[1], id2_2 = id2[2], id2_3 = id2[3];
        final double[] z = fz.clone();
        final int n = 5;
        return (x1, x2) -> {
            if (x1 < x1_0) x1 = x1_0; else if (x1 > x1_4) x1 = x1_4;
            if (x2 < x2_0) x2 = x2_0; else if (x2 > x2_4) x2 = x2_4;
            int i; double seg1, inv1;
            if (x1 <= x1_2) {
                if (x1 <= x1_1) { i = 0; seg1 = x1_0; inv1 = id1_0; }
                else             { i = 1; seg1 = x1_1; inv1 = id1_1; }
            } else {
                if (x1 <= x1_3) { i = 2; seg1 = x1_2; inv1 = id1_2; }
                else             { i = 3; seg1 = x1_3; inv1 = id1_3; }
            }
            int j; double seg2, inv2;
            if (x2 <= x2_2) {
                if (x2 <= x2_1) { j = 0; seg2 = x2_0; inv2 = id2_0; }
                else             { j = 1; seg2 = x2_1; inv2 = id2_1; }
            } else {
                if (x2 <= x2_3) { j = 2; seg2 = x2_2; inv2 = id2_2; }
                else             { j = 3; seg2 = x2_3; inv2 = id2_3; }
            }
            double s = (x1 - seg1) * inv1;
            double t = (x2 - seg2) * inv2;
            int b = i * n + j;
            return bilinearInterp(s, t, z[b], z[b + n], z[b + 1], z[b + n + 1]);
        };
    }

    private static GridBulkEvaluator compileBulkSquare5(double[] x1g, double[] x2g,
                                                         double[] id1, double[] id2, double[] fz) {
        final double x1_0 = x1g[0], x1_1 = x1g[1], x1_2 = x1g[2], x1_3 = x1g[3], x1_4 = x1g[4];
        final double x2_0 = x2g[0], x2_1 = x2g[1], x2_2 = x2g[2], x2_3 = x2g[3], x2_4 = x2g[4];
        final double id1_0 = id1[0], id1_1 = id1[1], id1_2 = id1[2], id1_3 = id1[3];
        final double id2_0 = id2[0], id2_1 = id2[1], id2_2 = id2[2], id2_3 = id2[3];
        final double[] z = fz.clone();
        final int n = 5;
        return (x1Values, x2Values, scores, count) -> {
            for (int idx = 0; idx < count; idx++) {
                double x1 = x1Values[idx];
                double x2 = x2Values[idx];
                if (x1 < x1_0) x1 = x1_0; else if (x1 > x1_4) x1 = x1_4;
                if (x2 < x2_0) x2 = x2_0; else if (x2 > x2_4) x2 = x2_4;
                int i; double seg1, inv1;
                if (x1 <= x1_2) {
                    if (x1 <= x1_1) { i = 0; seg1 = x1_0; inv1 = id1_0; }
                    else             { i = 1; seg1 = x1_1; inv1 = id1_1; }
                } else {
                    if (x1 <= x1_3) { i = 2; seg1 = x1_2; inv1 = id1_2; }
                    else             { i = 3; seg1 = x1_3; inv1 = id1_3; }
                }
                int j; double seg2, inv2;
                if (x2 <= x2_2) {
                    if (x2 <= x2_1) { j = 0; seg2 = x2_0; inv2 = id2_0; }
                    else             { j = 1; seg2 = x2_1; inv2 = id2_1; }
                } else {
                    if (x2 <= x2_3) { j = 2; seg2 = x2_2; inv2 = id2_2; }
                    else             { j = 3; seg2 = x2_3; inv2 = id2_3; }
                }
                double s = (x1 - seg1) * inv1;
                double t = (x2 - seg2) * inv2;
                int b = i * n + j;
                scores[idx] += bilinearInterp(s, t, z[b], z[b + n], z[b + 1], z[b + n + 1]);
            }
        };
    }

    // ── 6×6: 5×5 cells, binary-split + linear if/else per axis ──

    private static GridEvaluator compileSquare6(double[] x1g, double[] x2g,
                                                 double[] id1, double[] id2, double[] fz) {
        final double x1_0=x1g[0], x1_1=x1g[1], x1_2=x1g[2], x1_3=x1g[3], x1_4=x1g[4], x1_5=x1g[5];
        final double x2_0=x2g[0], x2_1=x2g[1], x2_2=x2g[2], x2_3=x2g[3], x2_4=x2g[4], x2_5=x2g[5];
        final double id1_0=id1[0], id1_1=id1[1], id1_2=id1[2], id1_3=id1[3], id1_4=id1[4];
        final double id2_0=id2[0], id2_1=id2[1], id2_2=id2[2], id2_3=id2[3], id2_4=id2[4];
        final double[] z = fz.clone();
        final int n = 6;
        return (x1, x2) -> {
            if (x1 < x1_0) x1 = x1_0; else if (x1 > x1_5) x1 = x1_5;
            if (x2 < x2_0) x2 = x2_0; else if (x2 > x2_5) x2 = x2_5;
            int i; double seg1, inv1;
            if (x1 <= x1_2) {
                if (x1 <= x1_1) { i = 0; seg1 = x1_0; inv1 = id1_0; }
                else             { i = 1; seg1 = x1_1; inv1 = id1_1; }
            } else if (x1 <= x1_3) { i = 2; seg1 = x1_2; inv1 = id1_2; }
            else {
                if (x1 <= x1_4) { i = 3; seg1 = x1_3; inv1 = id1_3; }
                else             { i = 4; seg1 = x1_4; inv1 = id1_4; }
            }
            int j; double seg2, inv2;
            if (x2 <= x2_2) {
                if (x2 <= x2_1) { j = 0; seg2 = x2_0; inv2 = id2_0; }
                else             { j = 1; seg2 = x2_1; inv2 = id2_1; }
            } else if (x2 <= x2_3) { j = 2; seg2 = x2_2; inv2 = id2_2; }
            else {
                if (x2 <= x2_4) { j = 3; seg2 = x2_3; inv2 = id2_3; }
                else             { j = 4; seg2 = x2_4; inv2 = id2_4; }
            }
            double s = (x1 - seg1) * inv1;
            double t = (x2 - seg2) * inv2;
            int b = i * n + j;
            return bilinearInterp(s, t, z[b], z[b + n], z[b + 1], z[b + n + 1]);
        };
    }

    private static GridBulkEvaluator compileBulkSquare6(double[] x1g, double[] x2g,
                                                         double[] id1, double[] id2, double[] fz) {
        final double x1_0=x1g[0], x1_1=x1g[1], x1_2=x1g[2], x1_3=x1g[3], x1_4=x1g[4], x1_5=x1g[5];
        final double x2_0=x2g[0], x2_1=x2g[1], x2_2=x2g[2], x2_3=x2g[3], x2_4=x2g[4], x2_5=x2g[5];
        final double id1_0=id1[0], id1_1=id1[1], id1_2=id1[2], id1_3=id1[3], id1_4=id1[4];
        final double id2_0=id2[0], id2_1=id2[1], id2_2=id2[2], id2_3=id2[3], id2_4=id2[4];
        final double[] z = fz.clone();
        final int n = 6;
        return (x1Values, x2Values, scores, count) -> {
            for (int idx = 0; idx < count; idx++) {
                double x1 = x1Values[idx];
                double x2 = x2Values[idx];
                if (x1 < x1_0) x1 = x1_0; else if (x1 > x1_5) x1 = x1_5;
                if (x2 < x2_0) x2 = x2_0; else if (x2 > x2_5) x2 = x2_5;
                int i; double seg1, inv1;
                if (x1 <= x1_2) {
                    if (x1 <= x1_1) { i = 0; seg1 = x1_0; inv1 = id1_0; }
                    else             { i = 1; seg1 = x1_1; inv1 = id1_1; }
                } else if (x1 <= x1_3) { i = 2; seg1 = x1_2; inv1 = id1_2; }
                else {
                    if (x1 <= x1_4) { i = 3; seg1 = x1_3; inv1 = id1_3; }
                    else             { i = 4; seg1 = x1_4; inv1 = id1_4; }
                }
                int j; double seg2, inv2;
                if (x2 <= x2_2) {
                    if (x2 <= x2_1) { j = 0; seg2 = x2_0; inv2 = id2_0; }
                    else             { j = 1; seg2 = x2_1; inv2 = id2_1; }
                } else if (x2 <= x2_3) { j = 2; seg2 = x2_2; inv2 = id2_2; }
                else {
                    if (x2 <= x2_4) { j = 3; seg2 = x2_3; inv2 = id2_3; }
                    else             { j = 4; seg2 = x2_4; inv2 = id2_4; }
                }
                double s = (x1 - seg1) * inv1;
                double t = (x2 - seg2) * inv2;
                int b = i * n + j;
                scores[idx] += bilinearInterp(s, t, z[b], z[b + n], z[b + 1], z[b + n + 1]);
            }
        };
    }

    // ── Fallback: precomputed inversions + binary search ──

    private static GridEvaluator compileFallback(double[] x1g, double[] x2g,
                                                  double[] id1, double[] id2,
                                                  double[] fz, int n1, int n2) {
        final double[] x1 = x1g.clone(), x2 = x2g.clone();
        final double[] invDx1 = id1.clone(), invDx2 = id2.clone();
        final double[] z = fz.clone();
        final double x1Lo = x1[0], x1Hi = x1[n1 - 1];
        final double x2Lo = x2[0], x2Hi = x2[n2 - 1];
        return (xv1, xv2) -> {
            if (xv1 < x1Lo) xv1 = x1Lo; else if (xv1 > x1Hi) xv1 = x1Hi;
            if (xv2 < x2Lo) xv2 = x2Lo; else if (xv2 > x2Hi) xv2 = x2Hi;
            int i = findSegment(x1, xv1, n1);
            int j = findSegment(x2, xv2, n2);
            double s = (xv1 - x1[i]) * invDx1[i];
            double t = (xv2 - x2[j]) * invDx2[j];
            int b = i * n2 + j;
            return bilinearInterp(s, t, z[b], z[b + n2], z[b + 1], z[b + n2 + 1]);
        };
    }

    private static GridBulkEvaluator compileBulkFallback(double[] x1g, double[] x2g,
                                                          double[] id1, double[] id2,
                                                          double[] fz, int n1, int n2) {
        final double[] x1 = x1g.clone(), x2 = x2g.clone();
        final double[] invDx1 = id1.clone(), invDx2 = id2.clone();
        final double[] z = fz.clone();
        final double x1Lo = x1[0], x1Hi = x1[n1 - 1];
        final double x2Lo = x2[0], x2Hi = x2[n2 - 1];
        final int cols = n2;
        return (x1Values, x2Values, scores, count) -> {
            for (int idx = 0; idx < count; idx++) {
                double xv1 = x1Values[idx];
                double xv2 = x2Values[idx];
                if (xv1 < x1Lo) xv1 = x1Lo; else if (xv1 > x1Hi) xv1 = x1Hi;
                if (xv2 < x2Lo) xv2 = x2Lo; else if (xv2 > x2Hi) xv2 = x2Hi;
                int i = findSegment(x1, xv1, n1);
                int j = findSegment(x2, xv2, n2);
                double s = (xv1 - x1[i]) * invDx1[i];
                double t = (xv2 - x2[j]) * invDx2[j];
                int b = i * cols + j;
                scores[idx] += bilinearInterp(s, t, z[b], z[b + cols], z[b + 1], z[b + cols + 1]);
            }
        };
    }

    /** Binary search for largest k where grid[k] <= x and k < n - 1. */
    private static int findSegment(double[] grid, double x, int n) {
        int lo = 0, hi = n - 2;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (grid[mid] <= x) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }

    public int x1Size() { return n1; }
    public int x2Size() { return n2; }
}
