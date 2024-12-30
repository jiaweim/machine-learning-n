package ml.weka.core.converters;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.File;

/**
 * Loads the data from the CSV file provided as first parameter and saves it
 * again to the CSV file provided as second parameter.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5457 $
 */
public class SaveDataToCsvFile {

  /**
   * Expects a filename as first and second parameter.
   *
   * @param args        the command-line parameters
   * @throws Exception  if something goes wrong
   */
  public static void main(String[] args) throws Exception {
    // output usage
    if (args.length != 2) {
      System.err.println("\nUsage: java SaveDataToCsvFile <input-file> <output-file>\n");
      System.exit(1);
    }

    System.out.println("\nReading from file " + args[0] + "...");
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File(args[0]));
    Instances data = loader.getDataSet();

    System.out.println("\nSaving to file " + args[1] + "...");
    CSVSaver saver = new CSVSaver();
    saver.setInstances(data);
    saver.setFile(new File(args[1]));
    saver.writeBatch();
  }
}
