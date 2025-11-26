package ml.djl;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 26 Nov 2025, 11:09 AM
 */
public class NDArrayTest {

    @Test
    void printDevice() {
        System.out.println(Device.cpu());
        System.out.println(Device.gpu());
        System.out.println(Device.gpu(1));
    }

    @Test
    void gpuCount() {
        System.out.println("GPU count: " + Engine.getInstance().getGpuCount());
    }

    @Test
    void arange() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(12f).reshape(3, 4);
            NDArray y = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4));

            var original = manager.zeros(y.getShape());
            var actual = original.addi(x);
            System.out.println(actual == original);
        }
    }
}
