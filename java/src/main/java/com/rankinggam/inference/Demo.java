package com.rankinggam.inference;

import java.io.File;
import java.util.Arrays;

/**
 * Demo: load a distilled GAM model and score/rerank documents.
 *
 * <p>Usage:
 * <pre>
 *   # Standard GAM scoring:
 *   python examples/demo_java_export.py --output gam_distilled.json
 *   java -cp ... com.rankinggam.inference.Demo gam_distilled.json
 *
 *   # Submodular reranking with diversity:
 *   python examples/demo_java_export.py --submodular --output submod.json
 *   java -cp ... com.rankinggam.inference.Demo submod.json --rerank
 * </pre>
 */
public class Demo {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: Demo <model.json> [options]");
            System.out.println();
            System.out.println("  model.json    Path to distilled model (from save_pwl_json)");
            System.out.println("  --rerank      Run submodular diversity reranking demo");
            System.out.println("  --num-features N  Feature count for random docs (default: auto)");
            System.out.println();
            System.out.println("Generate models with:");
            System.out.println("  python examples/demo_java_export.py --output gam.json");
            System.out.println("  python examples/demo_java_export.py --submodular --output submod.json");
            return;
        }

        String modelPath = args[0];
        boolean doRerank = false;
        int overrideFeatures = -1;
        for (int i = 1; i < args.length; i++) {
            if ("--rerank".equals(args[i])) doRerank = true;
            if ("--num-features".equals(args[i]) && i + 1 < args.length)
                overrideFeatures = Integer.parseInt(args[++i]);
        }

        if (doRerank) {
            demoSubmodularReranking(modelPath, overrideFeatures);
        } else {
            demoScoring(modelPath, overrideFeatures);
        }
    }

    // ── Standard GAM scoring demo ──

    static void demoScoring(String modelPath, int overrideFeatures) throws Exception {
        System.out.println("Loading model from " + modelPath + "...");

        DistilledGamModel model = DistilledGamLoader.fromJson(new File(modelPath));

        System.out.println("  Main effects: " + model.numMainEffects());
        System.out.println("  Interactions: " + model.numInteractions());
        System.out.printf("  Bias: %.6f%n", model.bias());

        int numFeatures = inferNumFeatures(model);
        if (overrideFeatures > 0) numFeatures = overrideFeatures;
        System.out.println("  Features: " + numFeatures);

        // Score some random documents
        int displaySize = 10;
        java.util.Random rng = new java.util.Random(42);
        double[][] docs = new double[displaySize][numFeatures];
        for (int i = 0; i < displaySize; i++)
            for (int j = 0; j < numFeatures; j++)
                docs[i][j] = rng.nextGaussian();

        System.out.println("\nScoring " + displaySize + " random documents:");
        double[] scores = model.score(docs);

        Integer[] indices = new Integer[displaySize];
        for (int i = 0; i < displaySize; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(scores[b], scores[a]));

        System.out.printf("  %-6s  %-12s  %s%n", "Rank", "Score", "Doc");
        for (int rank = 0; rank < displaySize; rank++) {
            int idx = indices[rank];
            System.out.printf("  %-6d  %12.6f  doc_%d%n", rank + 1, scores[idx], idx);
        }

        // Benchmarks
        System.out.println("\n--- Benchmarks: score(double[][] rowMajor) ---");
        for (int listSize : new int[]{10, 100, 1000, 10_000}) {
            benchmarkScoring(model, numFeatures, listSize);
        }

        System.out.println("\n--- Benchmarks: scoreColumnar(double[][] columns) ---");
        for (int listSize : new int[]{10, 100, 1000, 10_000}) {
            benchmarkColumnar(model, numFeatures, listSize);
        }
    }

    // ── Submodular reranking demo ──

    static void demoSubmodularReranking(String modelPath, int overrideFeatures) throws Exception {
        System.out.println("Loading submodular model from " + modelPath + "...");

        DistilledGamLoader.SubmodularModelData data =
                DistilledGamLoader.loadSubmodular(new File(modelPath));
        DistilledGamModel baseModel = data.baseModel;
        ConcavePwlFunction[] towers = data.diversityTowers;

        System.out.println("  Main effects:    " + baseModel.numMainEffects());
        System.out.println("  Interactions:    " + baseModel.numInteractions());
        System.out.println("  Diversity towers: " + towers.length);
        System.out.printf("  Bias: %.6f%n", baseModel.bias());

        int numFeatures = inferNumFeatures(baseModel);
        if (overrideFeatures > 0) numFeatures = overrideFeatures;
        System.out.println("  Features: " + numFeatures);

        if (towers.length == 0) {
            System.out.println("\nNo diversity towers found — use --submodular when exporting.");
            System.out.println("Falling back to standard scoring demo.");
            demoScoring(modelPath, overrideFeatures);
            return;
        }

        // Groupwise feature computer: use category on col 0, brand on col 1
        // (matches the default demo_java_export.py --submodular output)
        int catCol = 0, brandCol = 1;
        DefaultGroupwiseComputer.Spec[] specs = new DefaultGroupwiseComputer.Spec[towers.length];
        if (towers.length >= 1) specs[0] = DefaultGroupwiseComputer.Spec.novelty("category_novelty", catCol);
        if (towers.length >= 2) specs[1] = DefaultGroupwiseComputer.Spec.novelty("brand_novelty", brandCol);
        // Fill remaining with category_novelty if more towers exist
        for (int i = 2; i < towers.length; i++)
            specs[i] = DefaultGroupwiseComputer.Spec.novelty("category_novelty", catCol);
        DefaultGroupwiseComputer computer = new DefaultGroupwiseComputer(specs);

        SubmodularGamReranker reranker = new SubmodularGamReranker(baseModel, towers, computer);

        // Demo: 20 documents with 5 categories
        int displaySize = 20;
        int numCategories = 5;
        java.util.Random rng = new java.util.Random(42);
        double[][] docs = new double[displaySize][numFeatures];
        for (int i = 0; i < displaySize; i++) {
            docs[i][catCol] = i % numCategories;       // category 0..4
            docs[i][brandCol] = i % 3;                  // brand 0..2
            for (int j = 2; j < numFeatures; j++)
                docs[i][j] = rng.nextGaussian();
        }

        // Compare: base-only ranking vs diversity reranking
        double[] baseScores = baseModel.score(docs);
        Integer[] baseOrder = new Integer[displaySize];
        for (int i = 0; i < displaySize; i++) baseOrder[i] = i;
        Arrays.sort(baseOrder, (a, b) -> Double.compare(baseScores[b], baseScores[a]));

        System.out.println("\n=== Base GAM ranking (by relevance only) ===");
        System.out.printf("  %-6s  %-12s  %-8s  %-8s  %s%n",
                "Rank", "Score", "Cat", "Brand", "Doc");
        for (int rank = 0; rank < Math.min(10, displaySize); rank++) {
            int idx = baseOrder[rank];
            System.out.printf("  %-6d  %12.6f  cat_%.0f    brand_%.0f  doc_%d%n",
                    rank + 1, baseScores[idx],
                    docs[idx][catCol], docs[idx][brandCol], idx);
        }

        // Diversity reranking
        SubmodularGamReranker.RerankResult result =
                reranker.rerankWithScores(docs, 10, 10);

        System.out.println("\n=== Submodular reranking (relevance + diversity) ===");
        System.out.println("  (Minoux lazy greedy, maxEvalsPerPos=10)");
        System.out.printf("  %-6s  %-12s  %-8s  %-8s  %s%n",
                "Rank", "Score", "Cat", "Brand", "Doc");
        for (int rank = 0; rank < result.indices.length; rank++) {
            int idx = result.indices[rank];
            System.out.printf("  %-6d  %12.6f  cat_%.0f    brand_%.0f  doc_%d%n",
                    rank + 1, result.scores[rank],
                    docs[idx][catCol], docs[idx][brandCol], idx);
        }

        // Show diversity comparison
        java.util.Set<Double> baseCats = new java.util.HashSet<>();
        java.util.Set<Double> basebrands = new java.util.HashSet<>();
        for (int i = 0; i < Math.min(10, displaySize); i++) {
            baseCats.add(docs[baseOrder[i]][catCol]);
            basebrands.add(docs[baseOrder[i]][brandCol]);
        }
        java.util.Set<Double> divCats = new java.util.HashSet<>();
        java.util.Set<Double> divBrands = new java.util.HashSet<>();
        for (int idx : result.indices) {
            divCats.add(docs[idx][catCol]);
            divBrands.add(docs[idx][brandCol]);
        }

        System.out.println("\n=== Diversity comparison (top 10) ===");
        System.out.println("  Base ranking:  " + baseCats.size() + " categories, "
                + basebrands.size() + " brands");
        System.out.println("  Submod rerank: " + divCats.size() + " categories, "
                + divBrands.size() + " brands");

        // Benchmarks
        System.out.println("\n--- Benchmarks: SubmodularGamReranker ---");
        for (int listSize : new int[]{100, 1000, 10_000}) {
            benchmarkReranking(reranker, numFeatures, numCategories, listSize);
        }
    }

    // ── Benchmark helpers ──

    static void benchmarkScoring(DistilledGamModel model, int numFeatures, int listSize) {
        java.util.Random rng = new java.util.Random(42);
        double[][] bench = new double[listSize][numFeatures];
        for (int i = 0; i < listSize; i++)
            for (int j = 0; j < numFeatures; j++)
                bench[i][j] = rng.nextGaussian();

        int warmup = Math.max(100, 10000 / listSize);
        for (int i = 0; i < warmup; i++) model.score(bench);

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

    static void benchmarkColumnar(DistilledGamModel model, int numFeatures, int listSize) {
        java.util.Random rng = new java.util.Random(42);
        double[][] columns = new double[numFeatures][listSize];
        for (int j = 0; j < numFeatures; j++)
            for (int i = 0; i < listSize; i++)
                columns[j][i] = rng.nextGaussian();

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

    static void benchmarkReranking(SubmodularGamReranker reranker,
                                    int numFeatures, int numCategories, int listSize) {
        java.util.Random rng = new java.util.Random(42);
        double[][] features = new double[listSize][numFeatures];
        for (int i = 0; i < listSize; i++) {
            features[i][0] = i % numCategories;
            features[i][1] = i % 3;
            for (int j = 2; j < numFeatures; j++)
                features[i][j] = rng.nextGaussian();
        }

        int k = 40;

        // Warmup
        for (int w = 0; w < 3; w++) {
            reranker.rerank(features, k, 10);
            reranker.rerank(features, k);
        }

        int iterations = Math.max(10, 1000 / listSize);

        // Budget=10
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) reranker.rerank(features, k, 10);
        long elapsed = System.nanoTime() - start;
        double usWithBudget = elapsed / 1000.0 / iterations;

        // Unlimited (Minoux lazy greedy)
        start = System.nanoTime();
        for (int i = 0; i < iterations; i++) reranker.rerank(features, k);
        elapsed = System.nanoTime() - start;
        double usUnlimited = elapsed / 1000.0 / iterations;

        System.out.printf("  %,6d docs, k=%d: budget=10 %8.1f us, lazy-greedy %8.1f us%n",
                listSize, k, usWithBudget, usUnlimited);
    }

    static int inferNumFeatures(DistilledGamModel model) {
        int numFeatures = 0;
        for (DistilledGamModel.MainEffect me : model.mainEffects())
            numFeatures = Math.max(numFeatures, me.featureIndex() + 1);
        for (DistilledGamModel.Interaction ia : model.interactions())
            numFeatures = Math.max(numFeatures, Math.max(ia.feature1(), ia.feature2()) + 1);
        return numFeatures;
    }
}
