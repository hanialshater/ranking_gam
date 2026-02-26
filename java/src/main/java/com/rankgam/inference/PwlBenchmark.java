package com.rankgam.inference;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

/**
 * Latency benchmark for the normal (generic) PWL inference implementation.
 *
 * <p>Usage:
 * <pre>
 *   java com.rankgam.inference.PwlBenchmark model.json [batchSize] [listSize] [warmupIters] [benchIters]
 * </pre>
 *
 * <p>Defaults: batchSize=64, listSize=40, warmup=1000, bench=5000
 *
 * <p>Reports:
 * <ul>
 *   <li>Total time for the full benchmark loop</li>
 *   <li>Average latency per batch</li>
 *   <li>Average latency per query (single list)</li>
 *   <li>Average latency per document</li>
 *   <li>Throughput (documents/sec)</li>
 * </ul>
 *
 * <p>Compare these numbers against the compiled/code-generated variant to
 * measure the overhead of runtime interpolation and array lookups.
 */
public final class PwlBenchmark {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: PwlBenchmark <model.json> [batchSize] [listSize] [warmupIters] [benchIters]");
            System.exit(1);
        }

        Path modelPath = Paths.get(args[0]);
        int batchSize   = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        int listSize    = args.length > 2 ? Integer.parseInt(args[2]) : 40;
        int warmupIters = args.length > 3 ? Integer.parseInt(args[3]) : 1000;
        int benchIters  = args.length > 4 ? Integer.parseInt(args[4]) : 5000;

        System.out.println("Loading model from: " + modelPath);
        PwlModel model = PwlModelLoader.load(modelPath);
        PwlPredictor predictor = new PwlPredictor(model);

        int numFeatures = 0;
        for (PwlModel.MainEffect me : model.mainEffects) {
            numFeatures = Math.max(numFeatures, me.feature + 1);
        }
        for (PwlModel.Interaction ia : model.interactions) {
            numFeatures = Math.max(numFeatures, Math.max(ia.feature1, ia.feature2) + 1);
        }

        System.out.println("Model: " + model.mainEffects.size() + " main effects, "
                + model.interactions.size() + " interactions, "
                + numFeatures + " features");
        System.out.println("Benchmark: batch=" + batchSize + " list=" + listSize
                + " warmup=" + warmupIters + " bench=" + benchIters);
        System.out.println();

        // Generate random input data
        Random rng = new Random(42);
        double[][][] input = new double[batchSize][listSize][numFeatures];
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < listSize; d++) {
                for (int f = 0; f < numFeatures; f++) {
                    input[b][d][f] = rng.nextGaussian() * 2.0;
                }
            }
        }

        // Warmup
        System.out.println("Warming up (" + warmupIters + " iterations)...");
        for (int i = 0; i < warmupIters; i++) {
            predictor.predict(input);
        }

        // Benchmark
        System.out.println("Benchmarking (" + benchIters + " iterations)...");
        long totalDocs = (long) benchIters * batchSize * listSize;

        long startNs = System.nanoTime();
        for (int i = 0; i < benchIters; i++) {
            predictor.predict(input);
        }
        long elapsedNs = System.nanoTime() - startNs;

        double totalMs   = elapsedNs / 1_000_000.0;
        double perBatchUs = (elapsedNs / (double) benchIters) / 1_000.0;
        double perQueryUs = perBatchUs / batchSize;
        double perDocNs   = elapsedNs / (double) totalDocs;
        double docsPerSec = totalDocs / (totalMs / 1_000.0);

        System.out.println();
        System.out.println("=== Normal (generic) PWL inference results ===");
        System.out.printf("Total time:       %,.1f ms%n", totalMs);
        System.out.printf("Per batch:        %,.2f µs%n", perBatchUs);
        System.out.printf("Per query:        %,.2f µs%n", perQueryUs);
        System.out.printf("Per document:     %,.1f ns%n", perDocNs);
        System.out.printf("Throughput:       %,.0f docs/sec%n", docsPerSec);
        System.out.println();
        System.out.println("Compare these against the compiled/optimized variant.");
    }
}
