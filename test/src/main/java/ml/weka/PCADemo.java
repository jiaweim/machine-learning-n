package ml.weka;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.Standardize;

import java.net.URI;
import java.net.URL;
import java.util.Random;

/**
 * @author Jiawei Mao
 * @version 0.1.0
 * @since 13 Dec 2023, 9:23 PM
 */
public class PCADemo {

    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        URL url = new URI("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data").toURL();
        loader.setSource(url.openStream());
        Instances dataSet = loader.getDataSet();

        double percent = 70;
        int seed = 1;
        dataSet.randomize(new Random(seed));
        int trainSize = (int) Math.round(dataSet.numInstances() * percent / 100);
        int testSize = dataSet.numInstances() - trainSize;
        Instances train = new Instances(dataSet, 0, trainSize);
        Instances test = new Instances(dataSet, trainSize, testSize);

//        Standardize standardize = new Standardize();
//        standardize.setInputFormat(train);
//        train = Filter.useFilter(train, standardize);
//        test = Filter.useFilter(test, standardize);

        PrincipalComponents pca = new PrincipalComponents();
        pca.setOptions(Utils.splitOptions("-R 1"));
        pca.setInputFormat(train);
        Instances instances = Filter.useFilter(train, pca);
        System.out.println(new Instances(train, 0));
        System.out.println(new Instances(instances, 0));
    }
}
