package com.rankinggam.inference;

import java.io.File;
import java.util.Arrays;

/**
 * Demo: load a distilled GAM model and score documents.
 *
 * <p>Usage:
 * <pre>
 *   # First, export model from Python:
 *   python examples/demo_java_export.py --output gam_distilled.json
 *
 *   # Then run this demo:
 *   java -cp ... com.rankinggam.inference.Demo gam_distilled.json
 * </pre>
 */
public class Demo {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: Demo <model.json> [num_features]");
            System.out.println();
            System.out.println("  model.json    Path to distilled model (from save_pwl_json)");
            System.out.println("  num_features  Feature count for random docs (default: auto)");
            System.out.println();
            System.out.println("Generate a model with:");
            System.out.println("  python examples/demo_java_export.py --output gam_distilled.json");
            return;
        }

        String modelPath = args[0];
        System.out.println("Loading model from " + modelPath + "...");

        DistilledGamModel model = DistilledGamLoader.fromJson(new File(modelPath));

        System.out.println("  Main effects: " + model.numMainEffects());
        System.out.println("  Interactions: " + model.numInteractions());
        System.out.printf("  Bias: %.6f%n", model.bias());

        // Determine num features
        int numFeatures = 0;
        for (DistilledGamModel.MainEffect me : model.mainEffects()) {
            numFeatures = Math.max(numFeatures, me.featureIndex() + 1);
        }
        for (DistilledGamModel.Interaction ia : model.interactions()) {
            numFeatures = Math.max(numFeatures, Math.max(ia.feature1(), ia.feature2()) + 1);
        }
        if (args.length > 1) {
            numFeatures = Integer.parseInt(args[1]);
        }
        System.out.println("  Features: " + numFeatures);

        // Score some random documents (small list for display)
        int displaySize = 10;
        java.util.Random rng = new java.util.Random(42);
        double[][] docs = new double[displaySize][numFeatures];
        for (int i = 0; i < displaySize; i++) {
            for (int j = 0; j < numFeatures; j++) {
                docs[i][j] = rng.nextGaussian();
            }
        }

        System.out.println("\nScoring " + displaySize + " random documents:");
        double[] scores = model.score(docs);

        // Sort by score descending
        Integer[] indices = new Integer[displaySize];
        for (int i = 0; i < displaySize; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(scores[b], scores[a]));

        System.out.printf("  %-6s  %-12s  %s%n", "Rank", "Score", "Doc");
        for (int rank = 0; rank < displaySize; rank++) {
            int idx = indices[rank];
            System.out.printf("  %-6d  %12.6f  doc_%d%n", rank + 1, scores[idx], idx);
        }

        // Benchmark at multiple list sizes
        System.out.println("\n--- Benchmarks: score(double[][] rowMajor) ---");

        for (int listSize : new int[]{10, 100, 1000, 10_000}) {
            rng = new java.util.Random(42);
            double[][] bench = new double[listSize][numFeatures];
            for (int i = 0; i < listSize; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    bench[i][j] = rng.nextGaussian();
                }
            }

            // Warmup
            int warmup = Math.max(100, 10000 / listSize);
            for (int i = 0; i < warmup; i++) model.score(bench);

            // Timed iterations
            int iterations = Math.max(100, 100000 / listSize);
            long start = System.nanoTime();
            for (int i = 0; i < iterations; i++) model.score(bench);
            long elapsed = System.nanoTime() - start;

            double usPerList = elapsed / 1000.0 / iterations;
            double usPerDoc = usPerList / listSize;
            double docsPerSec = listSize * iterations * 1e9 / elapsed;
            System.out.printf("  %,6d docs: %8.1f us/list, %5.2f us/doc, %,.0f docs/sec%n",
                    listSize, usPerList, usPerDoc, docsPerSec);
        }

        // Benchmark columnar (SoA) input — no transpose needed
        System.out.println("\n--- Benchmarks: scoreColumnar(double[][] columns) ---");

        for (int listSize : new int[]{10, 100, 1000, 10_000}) {
            rng = new java.util.Random(42);
            double[][] columns = new double[numFeatures][listSize];
            for (int j = 0; j < numFeatures; j++) {
                for (int i = 0; i < listSize; i++) {
                    columns[j][i] = rng.nextGaussian();
                }
            }

            int warmup = Math.max(100, 10000 / listSize);
            for (int i = 0; i < warmup; i++) model.scoreColumnar(columns, listSize);

            int iterations = Math.max(100, 100000 / listSize);
            long start = System.nanoTime();
            for (int i = 0; i < iterations; i++) model.scoreColumnar(columns, listSize);
            long elapsed = System.nanoTime() - start;

            double usPerList = elapsed / 1000.0 / iterations;
            double usPerDoc = usPerList / listSize;
            double docsPerSec = listSize * iterations * 1e9 / elapsed;
            System.out.printf("  %,6d docs: %8.1f us/list, %5.2f us/doc, %,.0f docs/sec%n",
                    listSize, usPerList, usPerDoc, docsPerSec);
        }
    }
}
