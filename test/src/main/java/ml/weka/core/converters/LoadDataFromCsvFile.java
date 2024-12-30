package ml.weka.core.converters;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;

/**
 * Loads the data from the CSV file provided as first parameter.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class LoadDataFromCsvFile {

    /**
     * Expects a filename as first parameter.
     *
     * @param args the command-line parameters
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length != 1) {
            System.err.println("\nUsage: java LoadDataFromCsvFile <file>\n");
            System.exit(1);
        }

        System.out.println("\nReading file " + args[0] + "...");
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(args[0]));
        Instances data = loader.getDataSet();

        System.out.println("\nHeader of dataset:\n");
        System.out.println(new Instances(data, 0));
    }
}
