package ml.weka;

import org.junit.jupiter.api.Test;
import weka.core.WekaPackageManager;
import weka.core.packageManagement.Package;

/**
 * @author Jiawei Mao
 * @version 0.1.0
 * @since 12 Dec 2023, 8:16 PM
 */
public class PackagesDemo {

    @Test
    void loadPackages() {
        WekaPackageManager.loadPackages(false);
    }

    @Test
    void allPackages() throws Exception {
        for (Package p : WekaPackageManager.getAllPackages()) {
            System.out.println(p.getName() + "/" + p.getPackageMetaData().get("Version"));
        }

    }
}
