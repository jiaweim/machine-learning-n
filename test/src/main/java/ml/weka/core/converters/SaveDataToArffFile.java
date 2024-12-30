package ml.weka.core.converters;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

import java.io.File;

/**
 * Loads the data from the ARFF file provided as first parameter and saves it
 * again to the ARFF file provided as second parameter.  The input filename can
 * be either a local file or an URL.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class SaveDataToArffFile {

    /**
     * Expects a filename as first and second parameter.
     *
     * @param args the command-line parameters
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length != 2) {
            System.err.println("\nUsage: java SaveDataToArffFile <input-file|URL> <output-file>\n");
            System.exit(1);
        }

        System.out.println("\nReading from file " + args[0] + "...");
        ArffLoader loader = new ArffLoader();
        if (args[0].startsWith("http:") || args[0].startsWith("ftp:"))
            loader.setURL(args[0]);
        else
            loader.setSource(new File(args[0]));
        Instances data = loader.getDataSet();

        System.out.println("\nSaving to file " + args[1] + "...");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(args[1]));
        saver.writeBatch();
    }
}
