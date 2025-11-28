package ml.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 28 Nov 2025, 11:23 AM
 */
public class AlgeberaTest {
    @Test
    void scalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.create(3f);
            NDArray y = manager.create(2f);

            System.out.println(x.add(y));
            System.out.println(x.mul(y));
            System.out.println(x.div(y));
            System.out.println(x.pow(y));
        }
    }

    @Test
    void vector() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(4f);
            assertThat(x.size(0)).isEqualTo(4);
            assertThat(x.getShape()).isEqualTo(new Shape(4));
        }
    }

    @Test
    void matrix() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray A = manager.arange(20f).reshape(5, 4);
            System.out.println(A.transpose());
        }
    }

    @Test
    void matrix2() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray B = manager.create(new float[][]{{1, 2, 3}, {2, 0, 4}, {3, 4, 5}});
            System.out.println(B);
            System.out.println(B.eq(B.transpose()));
        }
    }

    @Test
    void ndarray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.arange(24f).reshape(2, 3, 4);
            System.out.println(X);
        }
    }

    @Test
    void add() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray A = manager.arange(20f).reshape(5, 4);
            System.out.println(A.sum());
        }
    }

    @Test
    void addScalar() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int a = 2;
            NDArray X = manager.arange(24f).reshape(2, 3, 4);
            System.out.println(X.mul(a).getShape());
        }
    }

    @Test
    void sum() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(4f);
            System.out.println(x);
            System.out.println(x.sum());
        }
    }

    @Test
    void sumAxis() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray A = manager.arange(20f).reshape(5, 4);
            NDArray aSumAxis0 = A.sum(new int[]{0});
            System.out.println(aSumAxis0);
            NDArray aSumAxis1 = A.sum(new int[]{1});
            System.out.println(aSumAxis1);

            System.out.println(A.sum(new int[]{0, 1}));
        }
    }

    @Test
    void mean() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray A = manager.arange(20f).reshape(5, 4);
            NDArray sumA = A.sum(new int[]{1}, true);
            System.out.println(A.cumSum(0));
        }
    }

    @Test
    void dotProduct() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray A = manager.arange(20f).reshape(5, 4);
            NDArray x = manager.arange(4f);
            NDArray y = manager.ones(new Shape(4));

            System.out.println(x.mul(y).sum());
        }
    }

    @Test
    void matrixVectorDot() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray A = manager.arange(20f).reshape(5, 4);
            NDArray x = manager.arange(4f);
            System.out.println(A.matMul(x));
        }
    }

    @Test
    void matrixMatrixDot() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray A = manager.arange(20f).reshape(5, 4);
            NDArray B = manager.ones(new Shape(4, 3));
            System.out.println(A.dot(B));
        }
    }

    @Test
    void norm() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray u = manager.create(new float[]{3, -4});
            System.out.println(l2Norm(manager.ones(new Shape(4, 9))));
        }
    }

    NDArray l2Norm(NDArray w) {
        return w.pow(2).sum().sqrt();
    }
}
