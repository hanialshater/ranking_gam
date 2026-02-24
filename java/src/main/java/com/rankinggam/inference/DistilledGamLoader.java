package com.rankinggam.inference;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads a distilled GAM model from JSON produced by Python's {@code save_pwl_json()}.
 *
 * <p>Expected JSON format:
 * <pre>
 * {
 *   "bias": 0.123,
 *   "main_effects": [
 *     {"feature": 0, "x": [0.0, 0.5, 1.0], "y": [0.1, 0.3, 0.8]},
 *     ...
 *   ],
 *   "interactions": [
 *     {"features": [3, 7], "x1_grid": [...], "x2_grid": [...], "z": [[...], ...]},
 *     ...
 *   ]
 * }
 * </pre>
 */
public final class DistilledGamLoader {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private DistilledGamLoader() {
    }

    /**
     * Load model from a JSON file path.
     */
    public static DistilledGamModel fromJson(String path) throws IOException {
        return fromJson(new File(path));
    }

    /**
     * Load model from a JSON file.
     */
    public static DistilledGamModel fromJson(File file) throws IOException {
        JsonNode root = MAPPER.readTree(file);
        return parseModel(root);
    }

    /**
     * Load model from an input stream (e.g., classpath resource).
     */
    public static DistilledGamModel fromJson(InputStream is) throws IOException {
        JsonNode root = MAPPER.readTree(is);
        return parseModel(root);
    }

    /**
     * Load model from a JSON string.
     */
    public static DistilledGamModel fromJsonString(String json) throws IOException {
        JsonNode root = MAPPER.readTree(json);
        return parseModel(root);
    }

    private static DistilledGamModel parseModel(JsonNode root) {
        double bias = root.get("bias").asDouble();

        // Main effects
        List<DistilledGamModel.MainEffect> mainEffects = new ArrayList<>();
        JsonNode meNode = root.get("main_effects");
        if (meNode != null && meNode.isArray()) {
            for (JsonNode me : meNode) {
                int featureIndex = me.get("feature").asInt();
                double[] x = toDoubleArray(me.get("x"));
                double[] y = toDoubleArray(me.get("y"));
                mainEffects.add(new DistilledGamModel.MainEffect(featureIndex, new PwlFunction(x, y)));
            }
        }

        // Interactions
        List<DistilledGamModel.Interaction> interactions = new ArrayList<>();
        JsonNode iaNode = root.get("interactions");
        if (iaNode != null && iaNode.isArray()) {
            for (JsonNode ia : iaNode) {
                JsonNode features = ia.get("features");
                int f1 = features.get(0).asInt();
                int f2 = features.get(1).asInt();
                double[] x1Grid = toDoubleArray(ia.get("x1_grid"));
                double[] x2Grid = toDoubleArray(ia.get("x2_grid"));
                double[][] z = toDouble2DArray(ia.get("z"));
                interactions.add(new DistilledGamModel.Interaction(
                        f1, f2, new BilinearGridFunction(x1Grid, x2Grid, z)));
            }
        }

        return new DistilledGamModel(bias, mainEffects, interactions);
    }

    private static double[] toDoubleArray(JsonNode node) {
        double[] arr = new double[node.size()];
        for (int i = 0; i < node.size(); i++) {
            arr[i] = node.get(i).asDouble();
        }
        return arr;
    }

    private static double[][] toDouble2DArray(JsonNode node) {
        double[][] arr = new double[node.size()][];
        for (int i = 0; i < node.size(); i++) {
            arr[i] = toDoubleArray(node.get(i));
        }
        return arr;
    }
}
