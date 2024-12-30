package ml.weka.core.converters;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Loads the data from the file provided as first parameter. The converter
 * is automatically chosen based on the file's extension.
 * The filename can be either a local file or an URL, if the actual converter
 * can also load data from URLs.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class LoadDataFromAnyFile {

    /**
     * Expects a filename as first parameter.
     *
     * @param args the command-line parameters
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length != 1) {
            System.err.println("\nUsage: java LoadDataFromAnyFile <file>\n");
            System.exit(1);
        }

        System.out.println("\nReading file " + args[0] + "...");
        Instances data = DataSource.read(args[0]);

        System.out.println("\nHeader of dataset:\n");
        System.out.println(new Instances(data, 0));
    }
}
