package com.rankinggam.inference;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BilinearGridFunctionTest {

    @Test
    void testCornerValues() {
        // 2x2 grid: z(x1, x2) where x1 in [0,1], x2 in [0,1]
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0.0, 1.0},
                new double[]{0.0, 1.0},
                new double[][]{{0.0, 1.0}, {2.0, 3.0}});

        assertEquals(0.0, f.evaluate(0.0, 0.0), 1e-9);
        assertEquals(1.0, f.evaluate(0.0, 1.0), 1e-9);
        assertEquals(2.0, f.evaluate(1.0, 0.0), 1e-9);
        assertEquals(3.0, f.evaluate(1.0, 1.0), 1e-9);
    }

    @Test
    void testBilinearCenter() {
        // At center (0.5, 0.5): average of all four corners
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0.0, 1.0},
                new double[]{0.0, 1.0},
                new double[][]{{0.0, 1.0}, {2.0, 3.0}});

        // (0.5)(0.5)*0 + 0.5*0.5*2 + 0.5*0.5*1 + 0.5*0.5*3 = 1.5
        assertEquals(1.5, f.evaluate(0.5, 0.5), 1e-9);
    }

    @Test
    void testEdgeInterpolation() {
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0.0, 1.0},
                new double[]{0.0, 1.0},
                new double[][]{{0.0, 1.0}, {2.0, 3.0}});

        // Along x1=0: linear interp between z(0,0)=0 and z(0,1)=1
        assertEquals(0.5, f.evaluate(0.0, 0.5), 1e-9);

        // Along x2=0: linear interp between z(0,0)=0 and z(1,0)=2
        assertEquals(1.0, f.evaluate(0.5, 0.0), 1e-9);
    }

    @Test
    void testClampOutOfRange() {
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0.0, 1.0},
                new double[]{0.0, 1.0},
                new double[][]{{0.0, 1.0}, {2.0, 3.0}});

        // Below range -> clamps to boundary
        assertEquals(0.0, f.evaluate(-1.0, -1.0), 1e-9);
        // Above range
        assertEquals(3.0, f.evaluate(2.0, 2.0), 1e-9);
    }

    @Test
    void testLargerGrid() {
        // 3x3 grid
        BilinearGridFunction f = new BilinearGridFunction(
                new double[]{0.0, 0.5, 1.0},
                new double[]{-1.0, 0.0, 1.0},
                new double[][]{
                        {0.0, 0.1, 0.2},
                        {0.1, 0.3, 0.5},
                        {0.2, 0.5, 1.0}
                });

        // At grid points
        assertEquals(0.0, f.evaluate(0.0, -1.0), 1e-9);
        assertEquals(0.3, f.evaluate(0.5, 0.0), 1e-9);
        assertEquals(1.0, f.evaluate(1.0, 1.0), 1e-9);
    }

    @Test
    void testInvalidInputs() {
        assertThrows(IllegalArgumentException.class,
                () -> new BilinearGridFunction(
                        new double[]{0.0}, // too short
                        new double[]{0.0, 1.0},
                        new double[][]{{0.0, 1.0}}));

        assertThrows(IllegalArgumentException.class,
                () -> new BilinearGridFunction(
                        new double[]{0.0, 1.0},
                        new double[]{0.0, 1.0},
                        new double[][]{{0.0}})); // wrong z dimensions
    }
}
