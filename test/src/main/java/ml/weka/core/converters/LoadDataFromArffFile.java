package ml.weka.core.converters;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

/**
 * Loads the data from the ARFF file provided as first parameter.
 * The filename can be either a local file or an URL.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class LoadDataFromArffFile {

    /**
     * Expects a filename as first parameter.
     *
     * @param args the command-line parameters
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length != 1) {
            System.err.println("\nUsage: java LoadDataFromArffFile <file|URL>\n");
            System.exit(1);
        }

        System.out.println("\nReading file " + args[0] + "...");
        ArffLoader loader = new ArffLoader();
        if (args[0].startsWith("http:") || args[0].startsWith("ftp:"))
            loader.setURL(args[0]);
        else
            loader.setSource(new File(args[0]));
        Instances data = loader.getDataSet();

        System.out.println("\nHeader of dataset:\n");
        System.out.println(new Instances(data, 0));
    }
}
