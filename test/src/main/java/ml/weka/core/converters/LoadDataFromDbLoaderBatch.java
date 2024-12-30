package ml.weka.core.converters;

import weka.core.Instances;
import weka.core.converters.DatabaseLoader;

/**
 * Loads data from a JDBC database using the weka.core.converters.DatabaseLoader
 * class. The data is loaded in batch mode.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class LoadDataFromDbLoaderBatch {

    /**
     * Expects no parameters.
     *
     * @param args the command-line parameters
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length != 0) {
            System.err.println("\nUsage: java LoadDataFromDbLoaderBatch\n");
            System.exit(1);
        }

        System.out.println("\nReading data...");
        DatabaseLoader loader = new DatabaseLoader();
        loader.setSource("jdbc_url", "the_user", "the_password");
        loader.setQuery("select * from whatsoever");
        Instances data = loader.getDataSet();

        System.out.println("\nHeader of dataset:\n");
        System.out.println(new Instances(data, 0));
    }
}
