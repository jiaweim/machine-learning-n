# HDBSCAN 聚类


## HDBSCAN Clustering

下面在一个 toy 数据集上展示如何使用 Tribuo 的 HDBSCAN 聚类包进行聚类和查找离群值。同时探讨如何可视化结果，并对新数据点进行预测。该实现细节可参考 [An Implementation of the HDBSCAN* Clustering Algorithm](https://www.mdpi.com/2076-3417/12/5/2405) .

### Setup

下面加载一些 jars，导入几个包。xchart jar 用于绘图，可以从 Maven 下载。

```java
%jars ./tribuo-clustering-hdbscan-4.3.0-jar-with-dependencies.jar
%jars ./xchart-3.8.1.jar
```

```java
import org.tribuo.*;
import org.tribuo.clustering.*;
import org.tribuo.clustering.hdbscan.*;
import org.tribuo.data.columnar.*;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.response.EmptyResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.knowm.xchart.*;
import org.knowm.xchart.style.markers.*;
import java.awt.Color;
import java.nio.file.Paths;
import java.util.*;
```

### 辅助绘图方法

下面声明几个用于可视化结果的方法。

```java
// A method to get a new instance of a chart, configured the same way each time
XYChart getNewXYChart(String title) {
    XYChart chart = new XYChartBuilder().width(600).height(400)
        .title(title).xAxisTitle("X").yAxisTitle("Y").build();
    chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);
    chart.getStyler().setChartTitleVisible(false);
    chart.getStyler().setLegendVisible(false);
    chart.getStyler().setMarkerSize(8);
    chart.getStyler().setPlotGridHorizontalLinesVisible(false);
    chart.getStyler().setPlotGridVerticalLinesVisible(false);
    return chart;
}
```

```java
// A method to add a set of (x,y) points to a chart
void addSeriesToChart(XYChart chart, List<Double> xList, List<Double> yList,
                      String seriesName, Color color, Marker marker) {
    XYSeries xYseries = chart.addSeries(seriesName,
            xList.stream().mapToDouble(Double::doubleValue).toArray(),
            yList.stream().mapToDouble(Double::doubleValue).toArray());
    xYseries.setMarkerColor(color);
    xYseries.setMarker(marker);
}
```
