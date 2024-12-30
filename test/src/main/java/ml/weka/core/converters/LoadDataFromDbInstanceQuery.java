package ml.weka.core.converters;

import weka.core.Instances;
import weka.experiment.InstanceQuery;

/**
 * Loads data from a JDBC database using the weka.experiment.InstanceQuery
 * class.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class LoadDataFromDbInstanceQuery {

    /**
     * Expects no parameters.
     *
     * @param args the command-line parameters
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length != 0) {
            System.err.println("\nUsage: java LoadDataFromDbInstanceQuery\n");
            System.exit(1);
        }

        System.out.println("\nReading data...");
        InstanceQuery query = new InstanceQuery();
        query.setDatabaseURL("jdbc_url");
        query.setUsername("the_user");
        query.setPassword("the_password");
        query.setQuery("select * from whatsoever");
        Instances data = query.retrieveInstances();

        System.out.println("\nHeader of dataset:\n");
        System.out.println(new Instances(data, 0));
    }
}
