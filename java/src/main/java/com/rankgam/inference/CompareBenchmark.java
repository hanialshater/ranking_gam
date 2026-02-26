package com.rankgam.inference;

import com.rankinggam.inference.BilinearGridFunction;
import com.rankinggam.inference.ConcavePwlFunction;
import com.rankinggam.inference.DefaultGroupwiseComputer;
import com.rankinggam.inference.DistilledGamModel;
import com.rankinggam.inference.PwlFunction;
import com.rankinggam.inference.FastSubmodularReranker;
import com.rankinggam.inference.SubmodularGamReranker;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Head-to-head latency benchmark: normal vs optimized PWL inference.
 *
 * <p>Sweeps across list sizes (10 → 10,000) to show how the optimized
 * code scales with document count. Tests three paths:
 * <ul>
 *   <li><b>Normal</b> ({@link PwlPredictor}): binary search + interp, row-major
 *   <li><b>Optimized row-major</b> ({@link DistilledGamModel#score}): compiled if/else, column extraction
 *   <li><b>Optimized columnar</b> ({@link DistilledGamModel#scoreColumnar}): compiled if/else, zero-copy
 * </ul>
 *
 * <p>Also benchmarks submodular diversity reranking when {@code --with-submodular}
 * is passed, using synthetic concave diversity towers.
 *
 * <p>Usage:
 * <pre>
 *   java com.rankgam.inference.CompareBenchmark model.json [--with-submodular]
 * </pre>
 */
public final class CompareBenchmark {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: CompareBenchmark <model.json> [--with-submodular]");
            System.exit(1);
        }

        Path modelPath = Paths.get(args[0]);
        boolean withSubmodular = false;
        for (int i = 1; i < args.length; i++)
            if ("--with-submodular".equals(args[i])) withSubmodular = true;

        // ── Load model ──
        PwlModel normalModel = PwlModelLoader.load(modelPath);
        PwlPredictor normalPredictor = new PwlPredictor(normalModel);
        DistilledGamModel optimizedModel = buildOptimized(normalModel);

        int numFeatures = 0;
        for (PwlModel.MainEffect me : normalModel.mainEffects)
            numFeatures = Math.max(numFeatures, me.feature + 1);
        for (PwlModel.Interaction ia : normalModel.interactions)
            numFeatures = Math.max(numFeatures, Math.max(ia.feature1, ia.feature2) + 1);

        System.out.println("Model: " + normalModel.mainEffects.size() + " main effects, "
                + normalModel.interactions.size() + " interactions, "
                + numFeatures + " features");

        // ── Correctness check ──
        Random rng = new Random(42);
        double[][] checkDocs = randomDocs(rng, 100, numFeatures);
        double[] normalScores = normalPredictor.predictQuery(checkDocs);
        double[] optimizedScores = optimizedModel.score(checkDocs);
        double maxDiff = 0;
        for (int i = 0; i < 100; i++)
            maxDiff = Math.max(maxDiff, Math.abs(normalScores[i] - optimizedScores[i]));
        System.out.printf("Correctness: max |normal - optimized| = %.2e%n%n", maxDiff);

        // ── Benchmark sweep ──
        int[] listSizes = {10, 100, 1_000, 10_000};

        System.out.printf("%-8s | %-28s | %-28s | %-28s | %s%n",
                "Docs", "Normal (generic)", "Optimized (row-major)", "Optimized (columnar)", "Speedup");
        System.out.printf("%-8s | %-12s %-14s | %-12s %-14s | %-12s %-14s | %s%n",
                "", "us/doc", "docs/sec", "us/doc", "docs/sec", "us/doc", "docs/sec", "col/norm");
        System.out.println("-".repeat(120));

        for (int listSize : listSizes) {
            rng = new Random(42);

            // Row-major input
            double[][] docs = randomDocs(rng, listSize, numFeatures);

            // Columnar input
            double[][] columns = new double[numFeatures][listSize];
            for (int j = 0; j < numFeatures; j++)
                for (int i = 0; i < listSize; i++)
                    columns[j][i] = docs[i][j];

            int warmup = Math.max(100, 10_000 / listSize);
            int iters  = Math.max(100, 100_000 / listSize);

            // Warmup all three
            for (int w = 0; w < warmup; w++) {
                normalPredictor.predictQuery(docs);
                optimizedModel.score(docs);
                optimizedModel.scoreColumnar(columns, listSize);
            }

            // Benchmark normal
            long t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) normalPredictor.predictQuery(docs);
            long normalNs = System.nanoTime() - t0;

            // Benchmark optimized row-major
            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) optimizedModel.score(docs);
            long optRowNs = System.nanoTime() - t0;

            // Benchmark optimized columnar
            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) optimizedModel.scoreColumnar(columns, listSize);
            long optColNs = System.nanoTime() - t0;

            long totalDocs = (long) iters * listSize;
            double normalUsDoc = normalNs / 1_000.0 / totalDocs;
            double optRowUsDoc = optRowNs / 1_000.0 / totalDocs;
            double optColUsDoc = optColNs / 1_000.0 / totalDocs;
            double normalDps   = totalDocs * 1e9 / normalNs;
            double optRowDps   = totalDocs * 1e9 / optRowNs;
            double optColDps   = totalDocs * 1e9 / optColNs;
            double speedup     = (double) normalNs / optColNs;

            System.out.printf("%,8d | %8.2f     %,14.0f | %8.2f     %,14.0f | %8.2f     %,14.0f | %.2fx%n",
                    listSize, normalUsDoc, normalDps, optRowUsDoc, optRowDps, optColUsDoc, optColDps, speedup);
        }

        // ── Submodular reranking benchmark ──
        if (withSubmodular) {
            System.out.println();
            System.out.println("=== Submodular Reranking: Original vs Fast ===");
            System.out.println("  2 diversity towers (category_novelty, brand_novelty), k=40");
            System.out.println("  Original: PriorityQueue<Candidate> + DefaultGroupwiseComputer (O(|S|) per eval)");
            System.out.println("  Fast: array heap + incremental novelty counters (O(1) per eval)");
            System.out.println();

            // Synthetic concave diversity towers: slopes=[0.8, 0.4, 0.2, 0.1]
            double[] knotEdges  = {0.0, 0.25, 0.5, 0.75};
            double[] knotWidths = {0.25, 0.25, 0.25, 0.25};
            double[] slopes     = {0.8, 0.4, 0.2, 0.1};
            ConcavePwlFunction tower1 = new ConcavePwlFunction(0.0, knotEdges, knotWidths, slopes, 0.0, 1.0);
            ConcavePwlFunction tower2 = new ConcavePwlFunction(0.0, knotEdges, knotWidths, slopes, 0.0, 1.0);
            ConcavePwlFunction[] towers = {tower1, tower2};

            int catCol = 0, brandCol = Math.min(1, numFeatures - 1);

            // Original reranker
            DefaultGroupwiseComputer.Spec[] specs = {
                DefaultGroupwiseComputer.Spec.novelty("category_novelty", catCol),
                DefaultGroupwiseComputer.Spec.novelty("brand_novelty", brandCol)
            };
            DefaultGroupwiseComputer computer = new DefaultGroupwiseComputer(specs);
            SubmodularGamReranker original = new SubmodularGamReranker(optimizedModel, towers, computer);

            // Fast reranker
            int[] noveltyCols = {catCol, brandCol};
            FastSubmodularReranker fast = new FastSubmodularReranker(optimizedModel, towers, noveltyCols);

            int k = 40;
            int numCategories = 5;

            System.out.printf("%-8s | %-22s | %-22s | %s%n",
                    "Docs", "Original (us/rerank)", "Fast (us/rerank)", "Speedup");
            System.out.println("-".repeat(72));

            for (int listSize : new int[]{100, 1_000, 10_000}) {
                rng = new Random(42);
                double[][] docs = new double[listSize][numFeatures];
                for (int i = 0; i < listSize; i++) {
                    docs[i][catCol] = i % numCategories;
                    docs[i][brandCol] = i % 3;
                    for (int j = 0; j < numFeatures; j++) {
                        if (j != catCol && j != brandCol)
                            docs[i][j] = rng.nextGaussian() * 2.0;
                    }
                }

                // Precompute base scores for fast reranker
                double[] baseScores = optimizedModel.score(docs);

                // Warmup
                for (int w = 0; w < 5; w++) {
                    original.rerank(docs, k);
                    fast.rerank(docs, baseScores, k, 0);
                }

                int iters = Math.max(10, 1_000 / listSize);

                // Benchmark original
                long t0 = System.nanoTime();
                for (int i = 0; i < iters; i++) original.rerank(docs, k);
                long origNs = System.nanoTime() - t0;
                double origUs = origNs / 1_000.0 / iters;

                // Benchmark fast (with precomputed base scores)
                t0 = System.nanoTime();
                for (int i = 0; i < iters; i++) fast.rerank(docs, baseScores, k, 0);
                long fastNs = System.nanoTime() - t0;
                double fastUs = fastNs / 1_000.0 / iters;

                double speedup = (double) origNs / fastNs;

                System.out.printf("%,8d | %,18.1f     | %,18.1f     | %.2fx%n",
                        listSize, origUs, fastUs, speedup);
            }
        }
    }

    private static double[][] randomDocs(Random rng, int listSize, int numFeatures) {
        double[][] docs = new double[listSize][numFeatures];
        for (int i = 0; i < listSize; i++)
            for (int j = 0; j < numFeatures; j++)
                docs[i][j] = rng.nextGaussian() * 2.0;
        return docs;
    }

    private static DistilledGamModel buildOptimized(PwlModel model) {
        List<DistilledGamModel.MainEffect> mains = new ArrayList<>();
        for (PwlModel.MainEffect me : model.mainEffects)
            mains.add(new DistilledGamModel.MainEffect(
                    me.feature, new PwlFunction(me.xKnots, me.yKnots)));

        List<DistilledGamModel.Interaction> inters = new ArrayList<>();
        for (PwlModel.Interaction ia : model.interactions)
            inters.add(new DistilledGamModel.Interaction(
                    ia.feature1, ia.feature2,
                    new BilinearGridFunction(ia.x1Grid, ia.x2Grid, ia.z)));

        return new DistilledGamModel(model.bias, mains, inters);
    }
}
