package ml.weka.core.converters;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Loads the data from the file provided as first parameter and saves it to a
 * file provided as second parameter. The converter is automatically chosen
 * based on the file's extension, this applies to the source and destination
 * file.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class SaveDataToAnyFile {

    /**
     * Expects a filename as first and second parameter.
     *
     * @param args the command-line parameters
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length != 2) {
            System.err.println("\nUsage: java SaveDataToAnyFile <input> <output>\n");
            System.exit(1);
        }

        System.out.println("\nReading from file " + args[0] + "...");
        Instances data = DataSource.read(args[0]);

        System.out.println("\nSaving to file " + args[1] + "...");
        DataSink.write(args[1], data);
    }
}
