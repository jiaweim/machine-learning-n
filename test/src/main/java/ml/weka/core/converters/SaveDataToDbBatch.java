package ml.weka.core.converters;

import weka.core.Instances;
import weka.core.converters.DatabaseLoader;
import weka.core.converters.DatabaseSaver;

/**
 * Loads data from a JDBC database using the
 * weka.core.converters.DatabaseLoader class and saves it to another JDBC
 * database using the weka.core.converters.DatabaseSaver class. The data is
 * loaded/saved in batch mode.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class SaveDataToDbBatch {

    /**
     * Expects no parameters.
     *
     * @param args the command-line parameters
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length != 0) {
            System.err.println("\nUsage: java SaveDataToDbBatch\n");
            System.exit(1);
        }

        System.out.println("\nReading data...");
        DatabaseLoader loader = new DatabaseLoader();
        loader.setSource("jdbc_url", "the_user", "the_password");
        loader.setQuery("select * from whatsoever");
        Instances data = loader.getDataSet();

        System.out.println("\nSaving data...");
        DatabaseSaver saver = new DatabaseSaver();
        saver.setDestination("jdbc_url", "the_user", "the_password");
        // we explicitly specify the table name here:
        saver.setTableName("whatsoever2");
        saver.setRelationForTableName(false);
        // or we could just update the name of the dataset:
        // saver.setRelationForTableName(true);
        // data.setRelationName("whatsoever2");
        saver.setInstances(data);
        saver.writeBatch();
    }
}
