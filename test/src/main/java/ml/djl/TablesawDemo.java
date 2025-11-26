package ml.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 26 Nov 2025, 6:02 PM
 */
public class TablesawDemo {

    static void createDataset() throws IOException {
        File file = new File("../data/");
        file.mkdir();

        String dataFile = "../data/house_tiny.csv";
        File f = new File(dataFile);
        f.createNewFile();

        try (FileWriter fw = new FileWriter(dataFile)) {
            fw.write("NumRooms,Alley,Price\n"); // Column names
            fw.write("NA,Pave,127500\n");  // Each row represents a data example
            fw.write("2,NA,106000\n");
            fw.write("4,NA,178100\n");
            fw.write("NA,NA,140000\n");
        }
    }

    static void read() {
        Table data = Table.read().file("../data/house_tiny.csv");

        Table inputs = data.create(data.columns());
        inputs.removeColumns("Price");
        Table outputs = data.selectColumns("Price");

        Column col = inputs.column("NumRooms");
        col.set(col.isMissing(), (int) inputs.nCol("NumRooms").mean());


        StringColumn alleyCol = (StringColumn) inputs.column("Alley");
        List<BooleanColumn> dummies = alleyCol.getDummies();
        inputs.removeColumns(alleyCol);
        inputs.addColumns(
                DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
                DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray())
        );
        System.out.println(inputs);


        try (NDManager nd = NDManager.newBaseManager()) {
            NDArray x = nd.create(inputs.as().doubleMatrix());
            NDArray y = nd.create(outputs.as().doubleMatrix());
            System.out.println(x);
            System.out.println(y);
        }

    }

    static void main() throws IOException {
        read();
    }
}
