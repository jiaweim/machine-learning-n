package ml.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 28 Nov 2025, 7:53 PM
 */
public class ProbabilityTest {

    @Test
    void touse() {
        float[] fairProbsArra = new float[6];
        Arrays.fill(fairProbsArra, 1f / 6);
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray fairProbs = manager.create(fairProbsArra);
            NDArray ndArray = manager.randomMultinomial(1000, fairProbs);
            NDArray counts = manager.randomMultinomial(10, fairProbs, new Shape(500));
            NDArray ndArray1 = counts.cumSum(0);
            System.out.println(ndArray.div(1000));
        }

    }
}
