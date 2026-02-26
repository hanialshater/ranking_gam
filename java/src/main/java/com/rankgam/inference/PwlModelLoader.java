package com.rankgam.inference;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads a distilled PWL model from JSON produced by the Python
 * {@code export_pwl_to_json()} utility.
 *
 * <p>Uses a minimal hand-rolled JSON parser — no external dependencies.
 * The JSON schema matches the Python dict format:
 * <pre>{@code
 * {
 *   "bias": 0.123,
 *   "main_effects": [
 *     {"feature": 0, "x": [0.0, 1.0, ...], "y": [0.1, 0.2, ...]},
 *     ...
 *   ],
 *   "interactions": [
 *     {"features": [3, 7], "x1_grid": [...], "x2_grid": [...], "z": [[...], ...]},
 *     ...
 *   ]
 * }
 * }</pre>
 */
public final class PwlModelLoader {

    private PwlModelLoader() {}

    /**
     * Load a PWL model from a JSON file.
     */
    public static PwlModel load(Path path) throws IOException {
        String json = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        return parse(json);
    }

    /**
     * Load a PWL model from an InputStream.
     */
    public static PwlModel load(InputStream in) throws IOException {
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        byte[] tmp = new byte[8192];
        int n;
        while ((n = in.read(tmp)) != -1) {
            buf.write(tmp, 0, n);
        }
        return parse(buf.toString("UTF-8"));
    }

    // ================================================================== //
    //  Minimal recursive-descent JSON parser                              //
    // ================================================================== //

    private static PwlModel parse(String json) {
        Parser p = new Parser(json);
        return p.parseModel();
    }

    private static final class Parser {
        private final String src;
        private int pos;

        Parser(String src) {
            this.src = src;
            this.pos = 0;
        }

        // -- top-level model ------------------------------------------------

        PwlModel parseModel() {
            double bias = 0;
            List<PwlModel.MainEffect> mains = new ArrayList<>();
            List<PwlModel.Interaction> inters = new ArrayList<>();

            skipWs();
            expect('{');
            while (peek() != '}') {
                String key = readString();
                skipWs();
                expect(':');
                skipWs();
                switch (key) {
                    case "bias":
                        bias = readNumber();
                        break;
                    case "main_effects":
                        mains = parseMainEffects();
                        break;
                    case "interactions":
                        inters = parseInteractions();
                        break;
                    default:
                        skipValue();
                        break;
                }
                skipWs();
                if (peek() == ',') advance();
            }
            expect('}');
            return new PwlModel(bias, mains, inters);
        }

        // -- main effects ---------------------------------------------------

        private List<PwlModel.MainEffect> parseMainEffects() {
            List<PwlModel.MainEffect> list = new ArrayList<>();
            expect('[');
            while (peek() != ']') {
                list.add(parseOneMainEffect());
                skipWs();
                if (peek() == ',') advance();
            }
            expect(']');
            return list;
        }

        private PwlModel.MainEffect parseOneMainEffect() {
            int feature = -1;
            double[] x = null, y = null;

            expect('{');
            while (peek() != '}') {
                String key = readString();
                skipWs();
                expect(':');
                skipWs();
                switch (key) {
                    case "feature":
                        feature = (int) readNumber();
                        break;
                    case "x":
                        x = readDoubleArray();
                        break;
                    case "y":
                        y = readDoubleArray();
                        break;
                    default:
                        skipValue();
                        break;
                }
                skipWs();
                if (peek() == ',') advance();
            }
            expect('}');
            return new PwlModel.MainEffect(feature, x, y);
        }

        // -- interactions ---------------------------------------------------

        private List<PwlModel.Interaction> parseInteractions() {
            List<PwlModel.Interaction> list = new ArrayList<>();
            expect('[');
            while (peek() != ']') {
                list.add(parseOneInteraction());
                skipWs();
                if (peek() == ',') advance();
            }
            expect(']');
            return list;
        }

        private PwlModel.Interaction parseOneInteraction() {
            int f1 = -1, f2 = -1;
            double[] x1Grid = null, x2Grid = null;
            double[][] z = null;

            expect('{');
            while (peek() != '}') {
                String key = readString();
                skipWs();
                expect(':');
                skipWs();
                switch (key) {
                    case "features":
                        double[] feats = readDoubleArray();
                        f1 = (int) feats[0];
                        f2 = (int) feats[1];
                        break;
                    case "x1_grid":
                        x1Grid = readDoubleArray();
                        break;
                    case "x2_grid":
                        x2Grid = readDoubleArray();
                        break;
                    case "z":
                        z = readDouble2DArray();
                        break;
                    default:
                        skipValue();
                        break;
                }
                skipWs();
                if (peek() == ',') advance();
            }
            expect('}');
            return new PwlModel.Interaction(f1, f2, x1Grid, x2Grid, z);
        }

        // -- primitives -----------------------------------------------------

        private double readNumber() {
            skipWs();
            int start = pos;
            if (peek() == '-') advance();
            while (pos < src.length() && (Character.isDigit(src.charAt(pos))
                    || src.charAt(pos) == '.' || src.charAt(pos) == 'e'
                    || src.charAt(pos) == 'E' || src.charAt(pos) == '+'
                    || src.charAt(pos) == '-') && pos > start) {
                // allow sign only right after e/E
                char c = src.charAt(pos);
                if ((c == '+' || c == '-') && pos > start) {
                    char prev = src.charAt(pos - 1);
                    if (prev != 'e' && prev != 'E') break;
                }
                advance();
            }
            // Handle case where we stopped at start+1 for negative numbers
            while (pos < src.length() && (Character.isDigit(src.charAt(pos))
                    || src.charAt(pos) == '.' || src.charAt(pos) == 'e'
                    || src.charAt(pos) == 'E')) {
                advance();
            }
            // Re-check for exponent sign
            if (pos < src.length() && (src.charAt(pos) == '+' || src.charAt(pos) == '-')) {
                char prev = src.charAt(pos - 1);
                if (prev == 'e' || prev == 'E') {
                    advance();
                    while (pos < src.length() && Character.isDigit(src.charAt(pos))) {
                        advance();
                    }
                }
            }
            return Double.parseDouble(src.substring(start, pos));
        }

        private String readString() {
            skipWs();
            expect('"');
            int start = pos;
            while (src.charAt(pos) != '"') {
                if (src.charAt(pos) == '\\') advance(); // skip escaped char
                advance();
            }
            String val = src.substring(start, pos);
            advance(); // closing "
            return val;
        }

        private double[] readDoubleArray() {
            List<Double> vals = new ArrayList<>();
            expect('[');
            while (peek() != ']') {
                vals.add(readNumber());
                skipWs();
                if (peek() == ',') advance();
            }
            expect(']');
            double[] arr = new double[vals.size()];
            for (int i = 0; i < arr.length; i++) arr[i] = vals.get(i);
            return arr;
        }

        private double[][] readDouble2DArray() {
            List<double[]> rows = new ArrayList<>();
            expect('[');
            while (peek() != ']') {
                rows.add(readDoubleArray());
                skipWs();
                if (peek() == ',') advance();
            }
            expect(']');
            return rows.toArray(new double[0][]);
        }

        /** Skip an arbitrary JSON value (used to ignore unknown keys). */
        private void skipValue() {
            skipWs();
            char c = peek();
            if (c == '"') {
                readString();
            } else if (c == '{') {
                advance();
                int depth = 1;
                while (depth > 0) {
                    char ch = src.charAt(pos);
                    if (ch == '{') depth++;
                    else if (ch == '}') depth--;
                    if (depth > 0) advance();
                }
                advance();
            } else if (c == '[') {
                advance();
                int depth = 1;
                while (depth > 0) {
                    char ch = src.charAt(pos);
                    if (ch == '[') depth++;
                    else if (ch == ']') depth--;
                    if (depth > 0) advance();
                }
                advance();
            } else if (c == 't' || c == 'f') {
                // true / false
                while (pos < src.length() && Character.isLetter(src.charAt(pos))) advance();
            } else if (c == 'n') {
                // null
                while (pos < src.length() && Character.isLetter(src.charAt(pos))) advance();
            } else {
                readNumber();
            }
        }

        // -- helpers --------------------------------------------------------

        private void skipWs() {
            while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) pos++;
        }

        private char peek() {
            skipWs();
            return src.charAt(pos);
        }

        private void advance() {
            pos++;
        }

        private void expect(char c) {
            skipWs();
            if (src.charAt(pos) != c) {
                throw new IllegalArgumentException(
                        "Expected '" + c + "' at position " + pos + ", got '" + src.charAt(pos) + "'");
            }
            advance();
        }
    }
}
