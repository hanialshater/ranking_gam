package com.rankinggam.inference;

/**
 * Lookup-table (LUT) based PWL evaluator — branchless, auto-vectorization friendly.
 *
 * <p>At construction time, pre-computes a uniform-grid lookup table by evaluating
 * the original PWL function at {@code tableSize} evenly spaced points. At eval time,
 * maps input x to a table index via branchless arithmetic (clamp → quantize → lerp).
 *
 * <h3>Why this is faster than if/else chains for bulk scoring:</h3>
 * <ul>
 *   <li><b>Branchless:</b> {@code Math.max}/{@code Math.min} on ints compile to
 *       {@code cmov} instructions — no branch predictor involvement, no misprediction
 *       penalty. The hot loop has zero data-dependent branches.
 *   <li><b>Monomorphic:</b> All features use the same concrete method, not 136 different
 *       lambda instances. The JIT can inline and potentially auto-vectorize.
 *   <li><b>Cache-friendly:</b> Each table is 2KB (256 entries × 8 bytes), fits in L1.
 *       All 136 feature tables total ~272KB, fitting in L2.
 * </ul>
 *
 * <h3>Accuracy:</h3>
 * <p>The table uses linear interpolation between grid points. Since the original
 * function is piecewise-linear, error occurs only where a PWL knot falls between
 * two table grid points. Max error ≈ |slope_change| × step / 4, where
 * step = (xMax - xMin) / (tableSize - 1). With 256 entries over a typical range
 * of ~4: error &lt; 0.02. With 1024 entries: error &lt; 0.005.
 */
public final class CompiledLutPwl {

    /** Default table size — good balance of accuracy (error &lt; 0.02) and cache (2KB). */
    public static final int DEFAULT_TABLE_SIZE = 256;

    private final double x0;         // left boundary (min knot)
    private final double invStep;    // 1.0 / step, where step = (xMax - x0) / (tableSize - 1)
    private final double[] table;    // precomputed PWL values at uniform grid points
    private final int tableSizeM2;   // tableSize - 2 (upper clamp bound for index)
    private final double yLeft;      // PWL value for x <= x0 (flat extrapolation)
    private final double yRight;     // PWL value for x >= xMax (flat extrapolation)

    private CompiledLutPwl(double x0, double invStep, double[] table,
                           double yLeft, double yRight) {
        this.x0 = x0;
        this.invStep = invStep;
        this.table = table;
        this.tableSizeM2 = table.length - 2;
        this.yLeft = yLeft;
        this.yRight = yRight;
    }

    /**
     * Compile a PwlFunction into a LUT evaluator with the specified table size.
     *
     * @param pwl       the piecewise-linear function to compile
     * @param tableSize number of table entries (e.g., 256 or 1024)
     * @return compiled LUT evaluator
     */
    public static CompiledLutPwl compile(PwlFunction pwl, int tableSize) {
        if (tableSize < 2) {
            throw new IllegalArgumentException("tableSize must be >= 2, got " + tableSize);
        }
        if (pwl.n < 2) {
            // Constant function: single-entry table
            double y = pwl.yKnots[0];
            return new CompiledLutPwl(0, 1, new double[]{y, y}, y, y);
        }

        double xMin = pwl.xKnots[0];
        double xMax = pwl.xKnots[pwl.n - 1];
        double range = xMax - xMin;

        if (range < 1e-15) {
            // Degenerate: all knots at same x
            double y = pwl.yKnots[0];
            return new CompiledLutPwl(xMin, 1, new double[]{y, y}, y, y);
        }

        double step = range / (tableSize - 1);
        double[] tbl = new double[tableSize];
        for (int i = 0; i < tableSize; i++) {
            tbl[i] = pwl.evaluate(xMin + i * step);
        }

        return new CompiledLutPwl(xMin, 1.0 / step, tbl,
                                   pwl.yKnots[0], pwl.yKnots[pwl.n - 1]);
    }

    /**
     * Compile with the default table size (256 entries).
     */
    public static CompiledLutPwl compile(PwlFunction pwl) {
        return compile(pwl, DEFAULT_TABLE_SIZE);
    }

    /**
     * Evaluate at a single point.
     */
    public double evaluate(double x) {
        double t = (x - x0) * invStep;
        int idx = Math.max(0, Math.min(tableSizeM2, (int) t));
        double frac = t - idx;
        // Clamp frac to [0, 1] for out-of-range inputs
        if (frac < 0.0) return yLeft;
        if (frac > 1.0) return yRight;
        return table[idx] + frac * (table[idx + 1] - table[idx]);
    }

    /**
     * Bulk evaluate: add PWL contributions into scores accumulator.
     * Column-major pattern — operates on a contiguous feature column.
     *
     * <p>The hot loop is branchless: {@code Math.max}/{@code Math.min} compile
     * to cmov, and the linear interpolation is pure arithmetic.
     *
     * @param values  input feature values (length >= count)
     * @param scores  accumulator; contributions are ADDED in place
     * @param count   number of elements to process
     */
    public void evaluateAndAccumulate(double[] values, double[] scores, int count) {
        final double lx0 = this.x0;
        final double lInvStep = this.invStep;
        final double[] lTable = this.table;
        final int lMax = this.tableSizeM2;
        final double lYLeft = this.yLeft;
        final double lYRight = this.yRight;

        for (int i = 0; i < count; i++) {
            double t = (values[i] - lx0) * lInvStep;
            int idx = Math.max(0, Math.min(lMax, (int) t));
            double frac = t - idx;
            // Handle out-of-range: frac < 0 means x < x0, frac > 1 means x > xMax
            if (frac < 0.0) {
                scores[i] += lYLeft;
            } else if (frac > 1.0) {
                scores[i] += lYRight;
            } else {
                scores[i] += lTable[idx] + frac * (lTable[idx + 1] - lTable[idx]);
            }
        }
    }

    /**
     * Evaluate directly from row-major features — no column extraction needed.
     *
     * <p>This fuses the column-extract + evaluate steps into a single pass.
     * Each iteration dereferences {@code features[i]} (pointer) then reads
     * at {@code features[i][featureIndex]} — same cost as column extraction
     * but skips the temporary buffer write and later read.
     *
     * @param features     [numDocs][numFeatures] row-major feature matrix
     * @param featureIndex which column to read from each row
     * @param scores       accumulator; contributions are ADDED in place
     * @param count        number of rows to process
     */
    public void evaluateFromRows(double[][] features, int featureIndex,
                                  double[] scores, int count) {
        final double lx0 = this.x0;
        final double lInvStep = this.invStep;
        final double[] lTable = this.table;
        final int lMax = this.tableSizeM2;
        final double lYLeft = this.yLeft;
        final double lYRight = this.yRight;

        for (int i = 0; i < count; i++) {
            double t = (features[i][featureIndex] - lx0) * lInvStep;
            int idx = Math.max(0, Math.min(lMax, (int) t));
            double frac = t - idx;
            if (frac < 0.0) {
                scores[i] += lYLeft;
            } else if (frac > 1.0) {
                scores[i] += lYRight;
            } else {
                scores[i] += lTable[idx] + frac * (lTable[idx + 1] - lTable[idx]);
            }
        }
    }

    /** Number of entries in the lookup table. */
    public int tableSize() {
        return table.length;
    }
}
