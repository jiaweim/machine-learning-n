package ml.djl;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

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
    void arange() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(12);
            assertThat(x.getDataType()).isSameAs(DataType.INT32);

            System.out.println(x);
        }
    }

    @Test
    void create(){
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(new Shape(3, 4));
            System.out.println(array);
        }
    }

    @Test
    void gpuCount() {
        System.out.println("GPU count: " + Engine.getInstance().getGpuCount());
    }


}
