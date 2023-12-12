package ml;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Generates an weka.core.Instances object with different attribute types.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class CreateInstances {

    /**
     * Generates the Instances object and outputs it in ARFF format to stdout.
     *
     * @param args ignored
     * @throws Exception if generation of instances fails
     */
    public static void main(String[] args) throws Exception {
        // 1. set up attributes
        ArrayList<Attribute> atts = new ArrayList<>();
        // - numeric
        atts.add(new Attribute("att1"));
        // - nominal
        ArrayList<String> attVals = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            attVals.add("val" + (i + 1));
        atts.add(new Attribute("att2", attVals));
        // - string
        atts.add(new Attribute("att3", (ArrayList<String>) null));
        // - date
        atts.add(new Attribute("att4", "yyyy-MM-dd"));
        // - relational
        ArrayList<Attribute> attsRel = new ArrayList<Attribute>();
        // -- numeric
        attsRel.add(new Attribute("att5.1"));
        // -- nominal
        ArrayList<String> attValsRel = new ArrayList<String>();
        for (int i = 0; i < 5; i++)
            attValsRel.add("val5." + (i + 1));
        attsRel.add(new Attribute("att5.2", attValsRel));
        Instances dataRel = new Instances("att5", attsRel, 0);
        atts.add(new Attribute("att5", dataRel, 0));

        // 2. create Instances object
        Instances data = new Instances("MyRelation", atts, 0);

        // 3. fill with data
        // first instance
        double[] vals = new double[data.numAttributes()];
        // - numeric
        vals[0] = Math.PI;
        // - nominal
        vals[1] = attVals.indexOf("val3");
        // - string
        vals[2] = data.attribute(2).addStringValue("This is a string!");
        // - date
        vals[3] = data.attribute(3).parseDate("2001-11-09");
        // - relational
        dataRel = new Instances(data.attribute(4).relation(), 0);
        // -- first instance
        double[] valsRel = new double[2];
        valsRel[0] = Math.PI + 1;
        valsRel[1] = attValsRel.indexOf("val5.3");
        dataRel.add(new DenseInstance(1.0, valsRel));
        // -- second instance
        valsRel = new double[2];
        valsRel[0] = Math.PI + 2;
        valsRel[1] = attValsRel.indexOf("val5.2");
        dataRel.add(new DenseInstance(1.0, valsRel));
        vals[4] = data.attribute(4).addRelation(dataRel);
        // add
        data.add(new DenseInstance(1.0, vals));

        // second instance
        vals = new double[data.numAttributes()];  // important: needs NEW array!
        // - numeric
        vals[0] = Math.E;
        // - nominal
        vals[1] = attVals.indexOf("val1");
        // - string
        vals[2] = data.attribute(2).addStringValue("And another one!");
        // - date
        vals[3] = data.attribute(3).parseDate("2000-12-01");
        // - relational
        dataRel = new Instances(data.attribute(4).relation(), 0);
        // -- first instance
        valsRel = new double[2];
        valsRel[0] = Math.E + 1;
        valsRel[1] = attValsRel.indexOf("val5.4");
        dataRel.add(new DenseInstance(1.0, valsRel));
        // -- second instance
        valsRel = new double[2];
        valsRel[0] = Math.E + 2;
        valsRel[1] = attValsRel.indexOf("val5.1");
        dataRel.add(new DenseInstance(1.0, valsRel));
        vals[4] = data.attribute(4).addRelation(dataRel);
        // add
        data.add(new DenseInstance(1.0, vals));

        // 4. output data
        System.out.println(data);
    }
}
