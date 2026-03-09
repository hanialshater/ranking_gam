package com.rankinggam.inference;

import java.util.Arrays;
import java.util.Random;

/**
 * Micro-benchmarks to measure the actual impact of inference optimizations.
 *
 * <p>Tests:
 * <ol>
 *   <li>CompiledPwlFunction vs PwlFunction (single-point and bulk)
 *   <li>Column-major bulk scoring vs row-by-row scoreDocument()
 *   <li>Columnar (zero-copy) vs row-major score()
 *   <li>BilinearGrid: single vs bulk evaluateAndAccumulate
 * </ol>
 *
 * <p>Run: {@code java -cp ... com.rankinggam.inference.OptimizationBenchmark}
 */
public class OptimizationBenchmark {

    // Warmup / measurement config
    private static final int WARMUP_MS = 2000;
    private static final int MEASURE_MS = 3000;

    public static void main(String[] args) {
        System.out.println("=== Inference Optimization Benchmarks ===");
        System.out.println("Warmup: " + WARMUP_MS + "ms, Measure: " + MEASURE_MS + "ms per trial");
        System.out.println();

        benchmarkPwlSinglePoint();
        benchmarkPwlBulk();
        benchmarkScoringLayouts();
        benchmarkBilinearSingleVsBulk();
    }

    // ── 1. CompiledPwl vs PwlFunction: single-point evaluation ──

    static void benchmarkPwlSinglePoint() {
        System.out.println("--- 1. Single-point PWL evaluation (CompiledPwl vs PwlFunction) ---");
        System.out.println("    Tests whether hand-unrolled if/else beats JIT-optimized linear scan.");
        System.out.println();

        // Test K=3 (typical distillation) and K=5
        for (int K : new int[]{3, 5}) {
            double[] xk = new double[K];
            double[] yk = new double[K];
            Random rng = new Random(42);
            for (int i = 0; i < K; i++) {
                xk[i] = (double) i / (K - 1);
                yk[i] = rng.nextDouble();
            }

            PwlFunction pwl = new PwlFunction(xk, yk);
            CompiledPwlFunction compiled = CompiledPwlFunction.compile(pwl);

            // Generate test inputs (mix of in-range, below, above)
            int N = 10_000;
            double[] inputs = new double[N];
            rng = new Random(123);
            for (int i = 0; i < N; i++) {
                inputs[i] = rng.nextDouble() * 1.4 - 0.2; // [-0.2, 1.2]
            }

            // Benchmark PwlFunction
            double opsPerSecPwl = benchmarkOps(() -> {
                double sink = 0;
                for (int i = 0; i < N; i++) sink += pwl.evaluate(inputs[i]);
                return sink;
            }, N);

            // Benchmark CompiledPwlFunction
            double opsPerSecCompiled = benchmarkOps(() -> {
                double sink = 0;
                for (int i = 0; i < N; i++) sink += compiled.evaluate(inputs[i]);
                return sink;
            }, N);

            double speedup = opsPerSecCompiled / opsPerSecPwl;
            System.out.printf("    K=%-2d  PwlFunction: %,.0f Mops/s  Compiled: %,.0f Mops/s  speedup: %.2fx%n",
                    K, opsPerSecPwl / 1e6, opsPerSecCompiled / 1e6, speedup);
        }
        System.out.println();
    }

    // ── 2. CompiledPwl vs PwlFunction: bulk evaluateAndAccumulate ──

    static void benchmarkPwlBulk() {
        System.out.println("--- 2. Bulk PWL evaluateAndAccumulate (Compiled vs PwlFunction) ---");
        System.out.println("    Tests the hot path used by column-major model scoring.");
        System.out.println();

        for (int K : new int[]{3, 5}) {
            double[] xk = new double[K];
            double[] yk = new double[K];
            Random rng = new Random(42);
            for (int i = 0; i < K; i++) {
                xk[i] = (double) i / (K - 1);
                yk[i] = rng.nextDouble();
            }

            PwlFunction pwl = new PwlFunction(xk, yk);
            CompiledPwlFunction compiled = CompiledPwlFunction.compile(pwl);

            for (int listSize : new int[]{40, 1000, 10_000}) {
                double[] values = new double[listSize];
                rng = new Random(123);
                for (int i = 0; i < listSize; i++) values[i] = rng.nextDouble() * 1.4 - 0.2;

                double[] scoresPwl = new double[listSize];
                double[] scoresCompiled = new double[listSize];

                // Benchmark PwlFunction bulk
                double opsPerSecPwl = benchmarkOps(() -> {
                    Arrays.fill(scoresPwl, 0);
                    pwl.evaluateAndAccumulate(values, scoresPwl, listSize);
                    return scoresPwl[0];
                }, listSize);

                // Benchmark CompiledPwlFunction bulk
                double opsPerSecCompiled = benchmarkOps(() -> {
                    Arrays.fill(scoresCompiled, 0);
                    compiled.evaluateAndAccumulate(values, scoresCompiled, listSize);
                    return scoresCompiled[0];
                }, listSize);

                double speedup = opsPerSecCompiled / opsPerSecPwl;
                System.out.printf("    K=%-2d  list=%,-6d  PwlFunction: %,.0f Mops/s  Compiled: %,.0f Mops/s  speedup: %.2fx%n",
                        K, listSize, opsPerSecPwl / 1e6, opsPerSecCompiled / 1e6, speedup);
            }
        }
        System.out.println();
    }

    // ── 3. Scoring layouts: row-by-row vs column-major vs columnar ──

    static void benchmarkScoringLayouts() {
        System.out.println("--- 3. Scoring layout comparison ---");
        System.out.println("    row-by-row: scoreDocument() in a loop");
        System.out.println("    column-major: score() with row-major input (internal column extraction)");
        System.out.println("    columnar: scoreColumnar() with pre-transposed columns (zero-copy)");
        System.out.println();

        // Build a realistic model: 10 main effects, K=4 knots each
        int numTowers = 10;
        int numFeatures = numTowers;
        java.util.List<DistilledGamModel.MainEffect> effects = new java.util.ArrayList<>();
        Random rng = new Random(42);
        for (int j = 0; j < numTowers; j++) {
            double[] xk = {0.0, 0.33, 0.67, 1.0};
            double[] yk = new double[4];
            for (int i = 0; i < 4; i++) yk[i] = rng.nextDouble() * 2 - 1;
            effects.add(new DistilledGamModel.MainEffect(j, new PwlFunction(xk, yk)));
        }
        DistilledGamModel model = new DistilledGamModel(0.5, effects, java.util.List.of());

        for (int listSize : new int[]{40, 200, 1000, 10_000}) {
            // Row-major features
            double[][] rowMajor = new double[listSize][numFeatures];
            rng = new Random(123);
            for (int i = 0; i < listSize; i++)
                for (int j = 0; j < numFeatures; j++)
                    rowMajor[i][j] = rng.nextDouble();

            // Column-major features (pre-transposed)
            double[][] columns = new double[numFeatures][listSize];
            for (int i = 0; i < listSize; i++)
                for (int j = 0; j < numFeatures; j++)
                    columns[j][i] = rowMajor[i][j];

            // Row-by-row: scoreDocument() in a loop
            double opsRowByRow = benchmarkOps(() -> {
                double sink = 0;
                for (int i = 0; i < listSize; i++)
                    sink += model.scoreDocument(rowMajor[i]);
                return sink;
            }, listSize);

            // Column-major: score() with internal transpose
            double opsColMajor = benchmarkOps(() -> {
                double[] s = model.score(rowMajor);
                return s[0];
            }, listSize);

            // Columnar: scoreColumnar() zero-copy
            double opsColumnar = benchmarkOps(() -> {
                double[] s = model.scoreColumnar(columns, listSize);
                return s[0];
            }, listSize);

            System.out.printf("    list=%,-6d  row-by-row: %,.0f Mops/s  col-major: %,.0f Mops/s (%.2fx)  columnar: %,.0f Mops/s (%.2fx)%n",
                    listSize,
                    opsRowByRow / 1e6,
                    opsColMajor / 1e6, opsColMajor / opsRowByRow,
                    opsColumnar / 1e6, opsColumnar / opsRowByRow);
        }
        System.out.println();
    }

    // ── 4. BilinearGrid: single vs bulk ──

    static void benchmarkBilinearSingleVsBulk() {
        System.out.println("--- 4. BilinearGrid single-point vs bulk evaluateAndAccumulate ---");
        System.out.println();

        // 5x5 grid (typical for GA2M interactions)
        int gridSize = 5;
        double[] x1Grid = new double[gridSize];
        double[] x2Grid = new double[gridSize];
        double[][] z = new double[gridSize][gridSize];
        Random rng = new Random(42);
        for (int i = 0; i < gridSize; i++) {
            x1Grid[i] = (double) i / (gridSize - 1);
            x2Grid[i] = (double) i / (gridSize - 1);
            for (int j = 0; j < gridSize; j++)
                z[i][j] = rng.nextDouble();
        }
        BilinearGridFunction grid = new BilinearGridFunction(x1Grid, x2Grid, z);

        for (int listSize : new int[]{40, 1000, 10_000}) {
            double[] x1Vals = new double[listSize];
            double[] x2Vals = new double[listSize];
            rng = new Random(123);
            for (int i = 0; i < listSize; i++) {
                x1Vals[i] = rng.nextDouble();
                x2Vals[i] = rng.nextDouble();
            }
            double[] scores = new double[listSize];

            // Single-point loop
            double opsSingle = benchmarkOps(() -> {
                double sink = 0;
                for (int i = 0; i < listSize; i++)
                    sink += grid.evaluate(x1Vals[i], x2Vals[i]);
                return sink;
            }, listSize);

            // Bulk
            double opsBulk = benchmarkOps(() -> {
                Arrays.fill(scores, 0);
                grid.evaluateAndAccumulate(x1Vals, x2Vals, scores, listSize);
                return scores[0];
            }, listSize);

            System.out.printf("    list=%,-6d  single-loop: %,.0f Mops/s  bulk: %,.0f Mops/s  speedup: %.2fx%n",
                    listSize, opsSingle / 1e6, opsBulk / 1e6, opsBulk / opsSingle);
        }
        System.out.println();
    }

    // ── Benchmark harness ──

    @FunctionalInterface
    interface BenchRunnable {
        double run();
    }

    /**
     * Warmup then measure, return operations per second.
     * @param opsPerCall how many "operations" (e.g. doc evaluations) happen per call
     */
    static double benchmarkOps(BenchRunnable fn, int opsPerCall) {
        // Warmup
        long deadline = System.nanoTime() + WARMUP_MS * 1_000_000L;
        double sink = 0;
        while (System.nanoTime() < deadline) {
            sink += fn.run();
        }

        // Measure
        long calls = 0;
        long start = System.nanoTime();
        deadline = start + MEASURE_MS * 1_000_000L;
        while (System.nanoTime() < deadline) {
            sink += fn.run();
            calls++;
        }
        long elapsed = System.nanoTime() - start;

        // Prevent dead code elimination
        if (sink == Double.NaN) System.out.println("never");

        return (double) calls * opsPerCall * 1e9 / elapsed;
    }
}
