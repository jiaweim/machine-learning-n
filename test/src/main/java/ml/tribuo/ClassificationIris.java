package ml.tribuo;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.oracle.labs.mlrg.olcut.config.json.JsonProvenanceModule;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.nio.file.Paths;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 21 Nov 2025, 6:00 PM
 */
public class ClassificationIris {

    public static void main(String[] args) throws IOException {
        // load data
        LabelFactory labelFactory = new LabelFactory();
        CSVLoader<Label> csvLoader = new CSVLoader<>(labelFactory);

        String[] irisHeaders = new String[]{"sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"};
        DataSource<Label> irisesSource = csvLoader.loadDataSource(Paths.get("D:\\data\\tribuo\\bezdekIris.data"), "species", irisHeaders);
        TrainTestSplitter<Label> irisSplitter = new TrainTestSplitter<>(irisesSource, 0.7, 1L);

        MutableDataset<Label> trainingDataset = new MutableDataset<>(irisSplitter.getTrain());
        MutableDataset<Label> testingDataset = new MutableDataset<>(irisSplitter.getTest());
        System.out.printf("Training data size = %d, number of features = %d, number of classes = %d%n",
                trainingDataset.size(), trainingDataset.getFeatureMap().size(), trainingDataset.getOutputInfo().size());
        System.out.printf("Testing data size = %d, number of features = %d, number of classes = %d%n",
                testingDataset.size(), testingDataset.getFeatureMap().size(), testingDataset.getOutputInfo().size());

        Trainer<Label> trainer = new LogisticRegressionTrainer();
        System.out.println(trainer.toString());

        Model<Label> irisModel = trainer.train(trainingDataset);

        LabelEvaluator evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(irisModel, testingDataset);
        System.out.println(evaluation.toString());
        System.out.println(evaluation.getConfusionMatrix().toString());

        ImmutableFeatureMap featureMap = irisModel.getFeatureIDMap();
        for (VariableInfo v : featureMap) {
            System.out.println(v.toString());
            System.out.println();
        }

        ModelProvenance provenance = irisModel.getProvenance();
        System.out.println(ProvenanceUtil.formattedProvenanceString(provenance.getDatasetProvenance().getSourceProvenance()));

        System.out.println(ProvenanceUtil.formattedProvenanceString(provenance.getTrainerProvenance()));

        ObjectMapper objMapper = new ObjectMapper();
        objMapper.registerModule(new JsonProvenanceModule());
        objMapper = objMapper.enable(SerializationFeature.INDENT_OUTPUT);

        String jsonProvenance = objMapper.writeValueAsString(ProvenanceUtil.marshalProvenance(provenance));
        System.out.println(jsonProvenance);

        System.out.println(irisModel.toString());

        String jsonEvaluationProvenance = objMapper.writeValueAsString(ProvenanceUtil.convertToMap(evaluation.getProvenance()));
        System.out.println(jsonEvaluationProvenance);
    }

}
