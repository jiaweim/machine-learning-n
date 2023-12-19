package ml.weka;

import org.junit.jupiter.api.Test;
import weka.classifiers.functions.LDA;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Utils;
import weka.estimators.MultivariateEstimator;
import weka.estimators.MultivariateGaussianEstimator;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * @author Jiawei Mao
 * @version 0.1.0
 * @since 14 Dec 2023, 3:42 PM
 */
public class LDATest {

    @Test
    void test() throws Exception {
        Attribute target = new Attribute("y", Arrays.asList("1", "2"));
        Attribute a = new Attribute("a");
        Attribute b = new Attribute("b");
        ArrayList<Attribute> attList = new ArrayList<>(2);
        attList.add(target);
        attList.add(a);
        attList.add(b);

        Instances dataset = new Instances("dataset", attList, 6);
        dataset.setClassIndex(0);

        dataset.add(new DenseInstance(1.0, new double[]{1., -1., -1.}));
        dataset.add(new DenseInstance(1.0, new double[]{1., -2, -1}));
        dataset.add(new DenseInstance(1.0, new double[]{1., -3, -2}));
        dataset.add(new DenseInstance(1.0, new double[]{2., 1, 1}));
        dataset.add(new DenseInstance(1.0, new double[]{2., 2, 1.}));
        dataset.add(new DenseInstance(1.0, new double[]{2., 3, 2}));

        LDA lda = new LDA();
        lda.setDebug(true);
        lda.buildClassifier(dataset);

        double label = lda.classifyInstance(new DenseInstance(1.0, new double[]{-1, 3, 2}));
        System.out.println(label);
    }

    public static void main(String[] args) {

        double[][] dataset1 = new double[4][1];
        dataset1[0][0] = 0.49;
        dataset1[1][0] = 0.46;
        dataset1[2][0] = 0.51;
        dataset1[3][0] = 0.55;

        MultivariateEstimator mv1 = new MultivariateGaussianEstimator();
        mv1.estimate(dataset1, new double[]{0.7, 0.2, 0.05, 0.05});

        System.err.println(mv1);

        double integral1 = 0;
        int numVals = 1000;
        for (int i = 0; i < numVals; i++) {
            double[] point = new double[1];
            point[0] = (i + 0.5) * (1.0 / numVals);
            double logdens = mv1.logDensity(point);
            if (!Double.isNaN(logdens)) {
                integral1 += Math.exp(logdens) * (1.0 / numVals);
            }
        }
        System.err.println("Approximate integral: " + integral1);

        double[][] dataset = new double[4][3];
        dataset[0][0] = 0.49;
        dataset[0][1] = 0.51;
        dataset[0][2] = 0.53;
        dataset[1][0] = 0.46;
        dataset[1][1] = 0.47;
        dataset[1][2] = 0.52;
        dataset[2][0] = 0.51;
        dataset[2][1] = 0.49;
        dataset[2][2] = 0.47;
        dataset[3][0] = 0.55;
        dataset[3][1] = 0.52;
        dataset[3][2] = 0.54;

        MultivariateEstimator mv = new MultivariateGaussianEstimator();
        mv.estimate(dataset, new double[]{2, 0.2, 0.05, 0.05});

        System.err.println(mv);

        double integral = 0;
        int numVals2 = 200;
        for (int i = 0; i < numVals2; i++) {
            for (int j = 0; j < numVals2; j++) {
                for (int k = 0; k < numVals2; k++) {
                    double[] point = new double[3];
                    point[0] = (i + 0.5) * (1.0 / numVals2);
                    point[1] = (j + 0.5) * (1.0 / numVals2);
                    point[2] = (k + 0.5) * (1.0 / numVals2);
                    double logdens = mv.logDensity(point);
                    if (!Double.isNaN(logdens)) {
                        integral += Math.exp(logdens) / (numVals2 * numVals2 * numVals2);
                    }
                }
            }
        }
        System.err.println("Approximate integral: " + integral);

        double[][] dataset3 = new double[5][3];
        dataset3[0][0] = 0.49;
        dataset3[0][1] = 0.51;
        dataset3[0][2] = 0.53;
        dataset3[4][0] = 0.49;
        dataset3[4][1] = 0.51;
        dataset3[4][2] = 0.53;
        dataset3[1][0] = 0.46;
        dataset3[1][1] = 0.47;
        dataset3[1][2] = 0.52;
        dataset3[2][0] = 0.51;
        dataset3[2][1] = 0.49;
        dataset3[2][2] = 0.47;
        dataset3[3][0] = 0.55;
        dataset3[3][1] = 0.52;
        dataset3[3][2] = 0.54;

        MultivariateEstimator mv3 = new MultivariateGaussianEstimator();
        mv3.estimate(dataset3, new double[]{1, 0.2, 0.05, 0.05, 1});

        System.err.println(mv3);

        double integral3 = 0;
        int numVals3 = 200;
        for (int i = 0; i < numVals3; i++) {
            for (int j = 0; j < numVals3; j++) {
                for (int k = 0; k < numVals3; k++) {
                    double[] point = new double[3];
                    point[0] = (i + 0.5) * (1.0 / numVals3);
                    point[1] = (j + 0.5) * (1.0 / numVals3);
                    point[2] = (k + 0.5) * (1.0 / numVals3);
                    double logdens = mv.logDensity(point);
                    if (!Double.isNaN(logdens)) {
                        integral3 += Math.exp(logdens) / (numVals3 * numVals3 * numVals3);
                    }
                }
            }
        }
        System.err.println("Approximate integral: " + integral3);

        double[][][] dataset4 = new double[2][][];
        dataset4[0] = new double[2][3];
        dataset4[1] = new double[3][3];
        dataset4[0][0][0] = 0.49;
        dataset4[0][0][1] = 0.51;
        dataset4[0][0][2] = 0.53;
        dataset4[0][1][0] = 0.49;
        dataset4[0][1][1] = 0.51;
        dataset4[0][1][2] = 0.53;
        dataset4[1][0][0] = 0.46;
        dataset4[1][0][1] = 0.47;
        dataset4[1][0][2] = 0.52;
        dataset4[1][1][0] = 0.51;
        dataset4[1][1][1] = 0.49;
        dataset4[1][1][2] = 0.47;
        dataset4[1][2][0] = 0.55;
        dataset4[1][2][1] = 0.52;
        dataset4[1][2][2] = 0.54;
        double[][] weights = new double[2][];
        weights[0] = new double[]{1, 3};
        weights[1] = new double[]{2, 1, 1};

        MultivariateGaussianEstimator mv4 = new MultivariateGaussianEstimator();
        mv4.estimatePooled(dataset4, weights);

        System.err.println(mv4);

        double integral4 = 0;
        int numVals4 = 200;
        for (int i = 0; i < numVals4; i++) {
            for (int j = 0; j < numVals4; j++) {
                for (int k = 0; k < numVals4; k++) {
                    double[] point = new double[3];
                    point[0] = (i + 0.5) * (1.0 / numVals4);
                    point[1] = (j + 0.5) * (1.0 / numVals4);
                    point[2] = (k + 0.5) * (1.0 / numVals4);
                    double logdens = mv.logDensity(point);
                    if (!Double.isNaN(logdens)) {
                        integral4 += Math.exp(logdens) / (numVals4 * numVals4 * numVals4);
                    }
                }
            }
        }
        System.err.println("Approximate integral: " + integral4);

        double[][][] dataset5 = new double[2][][];
        dataset5[0] = new double[4][3];
        dataset5[1] = new double[4][3];
        dataset5[0][0][0] = 0.49;
        dataset5[0][0][1] = 0.51;
        dataset5[0][0][2] = 0.53;
        dataset5[0][1][0] = 0.49;
        dataset5[0][1][1] = 0.51;
        dataset5[0][1][2] = 0.53;
        dataset5[0][2][0] = 0.49;
        dataset5[0][2][1] = 0.51;
        dataset5[0][2][2] = 0.53;
        dataset5[0][3][0] = 0.49;
        dataset5[0][3][1] = 0.51;
        dataset5[0][3][2] = 0.53;
        dataset5[1][0][0] = 0.46;
        dataset5[1][0][1] = 0.47;
        dataset5[1][0][2] = 0.52;
        dataset5[1][1][0] = 0.46;
        dataset5[1][1][1] = 0.47;
        dataset5[1][1][2] = 0.52;
        dataset5[1][2][0] = 0.51;
        dataset5[1][2][1] = 0.49;
        dataset5[1][2][2] = 0.47;
        dataset5[1][3][0] = 0.55;
        dataset5[1][3][1] = 0.52;
        dataset5[1][3][2] = 0.54;
        double[][] weights2 = new double[2][];
        weights2[0] = new double[]{1, 1, 1, 1};
        weights2[1] = new double[]{1, 1, 1, 1};

        MultivariateGaussianEstimator mv5 = new MultivariateGaussianEstimator();
        mv5.estimatePooled(dataset5, weights2);

        System.err.println(mv5);

        double integral5 = 0;
        int numVals5 = 200;
        for (int i = 0; i < numVals5; i++) {
            for (int j = 0; j < numVals5; j++) {
                for (int k = 0; k < numVals5; k++) {
                    double[] point = new double[3];
                    point[0] = (i + 0.5) * (1.0 / numVals5);
                    point[1] = (j + 0.5) * (1.0 / numVals5);
                    point[2] = (k + 0.5) * (1.0 / numVals5);
                    double logdens = mv.logDensity(point);
                    if (!Double.isNaN(logdens)) {
                        integral5 += Math.exp(logdens) / (numVals5 * numVals5 * numVals5);
                    }
                }
            }
        }
        System.err.println("Approximate integral: " + integral5);
    }
}
