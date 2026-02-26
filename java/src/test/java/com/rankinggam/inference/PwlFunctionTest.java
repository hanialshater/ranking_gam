package com.rankinggam.inference;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PwlFunctionTest {

    @Test
    void testLinearInterpolation() {
        // y = 2x on [0, 1]
        PwlFunction f = new PwlFunction(
                new double[]{0.0, 1.0},
                new double[]{0.0, 2.0});

        assertEquals(0.0, f.evaluate(0.0), 1e-9);
        assertEquals(1.0, f.evaluate(0.5), 1e-9);
        assertEquals(2.0, f.evaluate(1.0), 1e-9);
    }

    @Test
    void testClampBelowRange() {
        PwlFunction f = new PwlFunction(
                new double[]{1.0, 2.0},
                new double[]{10.0, 20.0});

        assertEquals(10.0, f.evaluate(0.0), 1e-9);
        assertEquals(10.0, f.evaluate(-100.0), 1e-9);
    }

    @Test
    void testClampAboveRange() {
        PwlFunction f = new PwlFunction(
                new double[]{1.0, 2.0},
                new double[]{10.0, 20.0});

        assertEquals(20.0, f.evaluate(3.0), 1e-9);
        assertEquals(20.0, f.evaluate(100.0), 1e-9);
    }

    @Test
    void testMultipleSegments() {
        PwlFunction f = new PwlFunction(
                new double[]{0.0, 1.0, 2.0, 3.0},
                new double[]{0.0, 1.0, 1.0, 2.0});

        // First segment: linear from 0->1
        assertEquals(0.5, f.evaluate(0.5), 1e-9);
        // Second segment: flat at 1
        assertEquals(1.0, f.evaluate(1.5), 1e-9);
        // Third segment: linear from 1->2
        assertEquals(1.5, f.evaluate(2.5), 1e-9);
    }

    @Test
    void testAtKnotPoints() {
        double[] x = {0.0, 0.5, 1.0};
        double[] y = {0.0, 0.3, 1.0};
        PwlFunction f = new PwlFunction(x, y);

        assertEquals(0.0, f.evaluate(0.0), 1e-9);
        assertEquals(0.3, f.evaluate(0.5), 1e-9);
        assertEquals(1.0, f.evaluate(1.0), 1e-9);
    }

    @Test
    void testBatchEvaluate() {
        PwlFunction f = new PwlFunction(
                new double[]{0.0, 1.0},
                new double[]{0.0, 2.0});

        double[] result = f.evaluate(new double[]{0.0, 0.25, 0.5, 0.75, 1.0});
        assertArrayEquals(new double[]{0.0, 0.5, 1.0, 1.5, 2.0}, result, 1e-9);
    }

    @Test
    void testSingleKnot() {
        PwlFunction f = new PwlFunction(
                new double[]{5.0},
                new double[]{3.0});

        assertEquals(3.0, f.evaluate(0.0), 1e-9);
        assertEquals(3.0, f.evaluate(5.0), 1e-9);
        assertEquals(3.0, f.evaluate(100.0), 1e-9);
    }

    @Test
    void testInvalidInputs() {
        assertThrows(IllegalArgumentException.class,
                () -> new PwlFunction(new double[]{}, new double[]{}));
        assertThrows(IllegalArgumentException.class,
                () -> new PwlFunction(new double[]{1.0}, new double[]{1.0, 2.0}));
    }

    @Test
    void testNumKnots() {
        PwlFunction f = new PwlFunction(
                new double[]{0.0, 0.5, 1.0},
                new double[]{0.0, 0.3, 1.0});
        assertEquals(3, f.numKnots());
    }
}
