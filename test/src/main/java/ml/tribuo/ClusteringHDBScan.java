package ml.tribuo;

import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.Marker;

import java.awt.*;
import java.util.List;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 22 Nov 2025, 15:15
 */
public class ClusteringHDBScan {

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

    // A method to add a set of (x,y) points to a chart
    void addSeriesToChart(XYChart chart, List<Double> xList, List<Double> yList,
                          String seriesName, Color color, Marker marker) {
        XYSeries xYseries = chart.addSeries(seriesName,
                xList.stream().mapToDouble(Double::doubleValue).toArray(),
                yList.stream().mapToDouble(Double::doubleValue).toArray());
        xYseries.setMarkerColor(color);
        xYseries.setMarker(marker);
    }

    public static void main(String[] args) {

    }
}
