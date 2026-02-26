package com.rankgam.inference;

import com.rankinggam.inference.BilinearGridFunction;
import com.rankinggam.inference.DistilledGamModel;
import com.rankinggam.inference.PwlFunction;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Head-to-head latency benchmark: normal vs optimized PWL inference.
 *
 * <ul>
 *   <li><b>Normal</b> ({@link PwlPredictor}): runtime binary search + interp per feature per doc
 *   <li><b>Optimized</b> ({@link DistilledGamModel}): compiled if/else chains, precomputed slopes,
 *       column-major bulk scoring
 * </ul>
 *
 * <p>Usage:
 * <pre>
 *   java com.rankgam.inference.CompareBenchmark model.json [batchSize] [listSize] [warmup] [iters]
 * </pre>
 */
public final class CompareBenchmark {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: CompareBenchmark <model.json> [batchSize] [listSize] [warmup] [iters]");
            System.exit(1);
        }

        Path modelPath = Paths.get(args[0]);
        int batchSize   = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        int listSize    = args.length > 2 ? Integer.parseInt(args[2]) : 40;
        int warmupIters = args.length > 3 ? Integer.parseInt(args[3]) : 2000;
        int benchIters  = args.length > 4 ? Integer.parseInt(args[4]) : 10000;

        // ── Load model (normal) ──
        PwlModel normalModel = PwlModelLoader.load(modelPath);
        PwlPredictor normalPredictor = new PwlPredictor(normalModel);

        // ── Build optimized model from the same parsed data ──
        DistilledGamModel optimizedModel = buildOptimized(normalModel);

        int numFeatures = 0;
        for (PwlModel.MainEffect me : normalModel.mainEffects) {
            numFeatures = Math.max(numFeatures, me.feature + 1);
        }
        for (PwlModel.Interaction ia : normalModel.interactions) {
            numFeatures = Math.max(numFeatures, Math.max(ia.feature1, ia.feature2) + 1);
        }

        System.out.println("Model: " + normalModel.mainEffects.size() + " main effects, "
                + normalModel.interactions.size() + " interactions, "
                + numFeatures + " features");
        System.out.println("Benchmark: batch=" + batchSize + " list=" + listSize
                + " warmup=" + warmupIters + " iters=" + benchIters);
        System.out.println();

        // ── Generate random input ──
        Random rng = new Random(42);
        double[][][] input = new double[batchSize][listSize][numFeatures];
        for (int b = 0; b < batchSize; b++)
            for (int d = 0; d < listSize; d++)
                for (int f = 0; f < numFeatures; f++)
                    input[b][d][f] = rng.nextGaussian() * 2.0;

        // ── Correctness check ──
        double[][] normalScores = normalPredictor.predict(input);
        double[][] optimizedScores = optimizedModel.scoreBatch(input);
        double maxDiff = 0;
        for (int b = 0; b < batchSize; b++)
            for (int d = 0; d < listSize; d++)
                maxDiff = Math.max(maxDiff, Math.abs(normalScores[b][d] - optimizedScores[b][d]));
        System.out.printf("Correctness check: max |normal - optimized| = %.2e%n%n", maxDiff);

        // ── Benchmark NORMAL ──
        System.out.println("Warming up NORMAL (" + warmupIters + " iters)...");
        for (int i = 0; i < warmupIters; i++) normalPredictor.predict(input);

        System.out.println("Benchmarking NORMAL (" + benchIters + " iters)...");
        long t0 = System.nanoTime();
        for (int i = 0; i < benchIters; i++) normalPredictor.predict(input);
        long normalNs = System.nanoTime() - t0;

        // ── Benchmark OPTIMIZED ──
        System.out.println("Warming up OPTIMIZED (" + warmupIters + " iters)...");
        for (int i = 0; i < warmupIters; i++) optimizedModel.scoreBatch(input);

        System.out.println("Benchmarking OPTIMIZED (" + benchIters + " iters)...");
        t0 = System.nanoTime();
        for (int i = 0; i < benchIters; i++) optimizedModel.scoreBatch(input);
        long optimizedNs = System.nanoTime() - t0;

        // ── Report ──
        long totalDocs = (long) benchIters * batchSize * listSize;
        printResults("NORMAL  (generic)", normalNs, benchIters, batchSize, listSize, totalDocs);
        printResults("OPTIMIZED (compiled)", optimizedNs, benchIters, batchSize, listSize, totalDocs);

        System.out.println();
        double speedup = (double) normalNs / optimizedNs;
        System.out.printf("Speedup: %.2fx%n", speedup);
    }

    private static void printResults(String label, long elapsedNs,
                                     int iters, int batchSize, int listSize, long totalDocs) {
        double totalMs   = elapsedNs / 1_000_000.0;
        double perBatchUs = (elapsedNs / (double) iters) / 1_000.0;
        double perQueryUs = perBatchUs / batchSize;
        double perDocNs   = elapsedNs / (double) totalDocs;
        double docsPerSec = totalDocs / (totalMs / 1_000.0);

        System.out.println();
        System.out.println("=== " + label + " ===");
        System.out.printf("  Total:      %,.1f ms%n", totalMs);
        System.out.printf("  Per batch:  %,.2f us%n", perBatchUs);
        System.out.printf("  Per query:  %,.2f us%n", perQueryUs);
        System.out.printf("  Per doc:    %,.1f ns%n", perDocNs);
        System.out.printf("  Throughput: %,.0f docs/sec%n", docsPerSec);
    }

    /** Build DistilledGamModel from the parsed PwlModel (no Jackson needed). */
    private static DistilledGamModel buildOptimized(PwlModel model) {
        List<DistilledGamModel.MainEffect> mains = new ArrayList<>();
        for (PwlModel.MainEffect me : model.mainEffects) {
            mains.add(new DistilledGamModel.MainEffect(
                    me.feature, new PwlFunction(me.xKnots, me.yKnots)));
        }

        List<DistilledGamModel.Interaction> inters = new ArrayList<>();
        for (PwlModel.Interaction ia : model.interactions) {
            inters.add(new DistilledGamModel.Interaction(
                    ia.feature1, ia.feature2,
                    new BilinearGridFunction(ia.x1Grid, ia.x2Grid, ia.z)));
        }

        return new DistilledGamModel(model.bias, mains, inters);
    }
}
