package ml.djl;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import org.junit.jupiter.api.Test;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.util.function.Function;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 28 Nov 2025, 4:14 PM
 */
public class CalculusTest {
    @Test
    void f() {
        Function<Double, Double> f = x -> 3 * Math.pow(x, 2) - 4 * x;

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.arange(0f, 3f, 0.1f, DataType.FLOAT64);
            double[] x = X.toDoubleArray();

            double[] fx = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                fx[i] = f.apply(x[i]);
            }

            double[] fg = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                fg[i] = 2 * x[i] - 3;
            }

            Figure figure = plotLineAndSegment(x, fx, fg,
                    "f(x)", "Tangent line(x=1)",
                    "x", "f(x)", 700, 500);
            Plot.show(figure);
        }
    }

    Double limit(Function<Double, Double> f, double x, double h) {
        return (f.apply(x + h) - f.apply(x)) / h;
    }

    public Figure plotLineAndSegment(double[] x, double[] y, double[] segment,
            String trace1Name, String trace2Name,
            String xLabel, String yLabel,
            int width, int height) {
        ScatterTrace trace = ScatterTrace.builder(x, y)
                .mode(ScatterTrace.Mode.LINE)
                .name(trace1Name)
                .build();

        ScatterTrace trace2 = ScatterTrace.builder(x, segment)
                .mode(ScatterTrace.Mode.LINE)
                .name(trace2Name)
                .build();

        Layout layout = Layout.builder()
                .height(height)
                .width(width)
                .showLegend(true)
                .xAxis(Axis.builder().title(xLabel).build())
                .yAxis(Axis.builder().title(yLabel).build())
                .build();

        return new Figure(layout, trace, trace2);
    }

    @Test
    void diff() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(4f);
            // 表示要计算 x 的梯度，为其分配内存
            x.setRequiresGradient(true);
            // 计算梯度后，通过 getGradient 获取梯度值，梯度初始化为 0
            System.out.println(x.getGradient());

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray y = x.mul(x);
                NDArray u = y.stopGradient();
                NDArray z = u.mul(x);
                gc.backward(z);
                System.out.println(x.getGradient().eq(u));
            }


            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray y = x.mul(x);
                y = x.mul(x);
                gc.backward(y);
                System.out.println(x.getGradient().eq(x.mul(2)));
            }

        }
    }

    public NDArray f(NDArray a) {
        NDArray b = a.mul(2);
        while (b.norm().getFloat() < 1000) {
            b = b.mul(2);
        }
        NDArray c;
        if (b.sum().getFloat() > 0) {
            c = b;
        } else {
            c = b.mul(100);
        }
        return c;
    }

    @Test
    void flowGrad() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray a = manager.randomNormal(new Shape(1));
            a.setRequiresGradient(true);
            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray d = f(a);
                gc.backward(d);
                System.out.println(a.getGradient().eq(d.div(a)));
            }
        }

    }
}
