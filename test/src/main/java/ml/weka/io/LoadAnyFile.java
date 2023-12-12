package ml.weka.io;


import org.junit.jupiter.api.Test;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.IOException;

/**
 * @author Jiawei Mao
 * @version 0.1.0
 * @since 12 Dec 2023, 8:47 PM
 */
public class LoadAnyFile {

    public static void main(String[] args) throws Exception {
        String file = "";
        Instances instances = DataSource.read(file);

    }

    @Test
    void readArff() throws IOException {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("C:\\Program Files\\Weka-3-9-6\\data\\iris.arff"));
        Instances dataSet = loader.getDataSet();
//        System.out.println(new Instances(dataSet, 0));
        System.out.println(dataSet);
    }

    @Test
    void readCsv() throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("D:\\iris.csv"));
        Instances dataSet = loader.getDataSet();
        System.out.println("\nHeader of dataset:\n");
        System.out.println(new Instances(dataSet, 0));
    }
}
