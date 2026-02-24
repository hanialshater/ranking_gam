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

        // Score some random documents
        int listSize = 10;
        java.util.Random rng = new java.util.Random(42);
        double[][] docs = new double[listSize][numFeatures];
        for (int i = 0; i < listSize; i++) {
            for (int j = 0; j < numFeatures; j++) {
                docs[i][j] = rng.nextGaussian();
            }
        }

        System.out.println("\nScoring " + listSize + " random documents:");
        double[] scores = model.score(docs);

        // Sort by score descending
        Integer[] indices = new Integer[listSize];
        for (int i = 0; i < listSize; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(scores[b], scores[a]));

        System.out.printf("  %-6s  %-12s  %s%n", "Rank", "Score", "Doc");
        for (int rank = 0; rank < listSize; rank++) {
            int idx = indices[rank];
            System.out.printf("  %-6d  %12.6f  doc_%d%n", rank + 1, scores[idx], idx);
        }

        // Benchmark
        int warmup = 1000;
        int iterations = 100_000;
        for (int i = 0; i < warmup; i++) model.score(docs);

        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) model.score(docs);
        long elapsed = System.nanoTime() - start;

        double usPerList = elapsed / 1000.0 / iterations;
        double usPerDoc = usPerList / listSize;
        System.out.printf("%nBenchmark (%d features, %d docs/list, %d iterations):%n",
                numFeatures, listSize, iterations);
        System.out.printf("  %.1f us/list, %.2f us/doc%n", usPerList, usPerDoc);
    }
}
