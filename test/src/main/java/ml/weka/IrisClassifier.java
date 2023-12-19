package ml.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.ArrayList;
import java.util.Random;


/**
 * @author Jiawei Mao
 * @version 0.1.0
 * @since 15 Dec 2023, 11:34 AM
 */
public class IrisClassifier {

    public static Instances loadDataset(String path) throws Exception {
        Instances dataset = ConverterUtils.DataSource.read(path);
        if (dataset.classIndex() == -1)
            dataset.setClassIndex(dataset.numAttributes() - 1);
        return dataset;
    }

    public static Classifier buildClassifier(Instances dataset) throws Exception {
        MultilayerPerceptron m = new MultilayerPerceptron();
        m.buildClassifier(dataset);
        return m;
    }

    public static String evaluateModel(Classifier model, Instances trainSataset, Instances testDataset) throws Exception {
        Evaluation evaluation = new Evaluation(trainSataset);
        evaluation.evaluateModel(model, testDataset);
        return evaluation.toSummaryString("", true);
    }

    public static void saveModel(Classifier model, String path) throws Exception {
        SerializationHelper.write(path, model);
    }

    public static Instances createInstance(Instances dataset, double petalLength, double petalWidth, double result) {
        dataset.clear();
        double[] value1 = new double[]{petalLength, petalWidth, 0};
        dataset.add(new DenseInstance(1, value1));
        return dataset;
    }

    public static String classify(Instances dataset, String path, ArrayList<String> classes) throws Exception {
        String result = "Not classified!";
        Classifier cls = (MultilayerPerceptron) SerializationHelper.read(path);
        double v = cls.classifyInstance(dataset.firstInstance());
        result = classes.get((int) v);
        return result;
    }

    public static void main(String[] args) throws Exception {

        String dataPath = "C:\\Program Files\\Weka-3-9-6\\data\\iris.2D.arff";
        Instances ds1 = loadDataset(dataPath);

        Filter filter = new Normalize();

        int trainSize = (int) Math.round(ds1.numInstances() * 0.8);
        int testSize = ds1.numInstances() - trainSize;

        ds1.randomize(new Random(1));
        filter.setInputFormat(ds1);

        Instances normalizedDataset = Filter.useFilter(ds1, filter);

        Instances trainDataset = new Instances(normalizedDataset, 0, trainSize);
        Instances testDataset = new Instances(normalizedDataset, trainSize, testSize);

        MultilayerPerceptron classifier = (MultilayerPerceptron) buildClassifier(trainDataset);
        String evalSummary = evaluateModel(classifier, trainDataset, testDataset);
        System.out.println(evalSummary);

        String modelPath = "D:\\data\\test\\model.bin";
        saveModel(classifier, modelPath);


        // 创建分类数据
        Attribute petalLength = new Attribute("petalLength");
        Attribute petalWidth = new Attribute("petalWidth");

        ArrayList<Attribute> attList = new ArrayList<>(3);
        attList.add(petalLength);
        attList.add(petalWidth);

        ArrayList<String> classes = new ArrayList<>(3);
        classes.add("Iris-setosa");
        classes.add("Iris-versicolor");
        classes.add("Iris-virginica");
        Attribute cls = new Attribute("class", classes);

        attList.add(cls);

        Instances dataset = new Instances("TestDataset", attList, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);

        Instances instance = createInstance(dataset, 1.6, 0.2, 0);
        Instances d3 = Filter.useFilter(instance, filter);

        String className = classify(d3, modelPath, classes);
        System.out.println("The class name for the instance with petalLength=1.6 and petalWidth=0.2 is " + className);

    }

}
