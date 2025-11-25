package ml.djl;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 25 Nov 2025, 4:18 PM
 */
public class FirstNetwork {
    public static void main(String[] args) throws IOException, TranslateException {
        int batchSize = 32;
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        mnist.prepare(new ProgressBar());

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(new Mlp(28 * 28, 10, new int[]{128, 64}));

            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(1, 28 * 28));

            int epoch = 2;
            EasyTrain.fit(trainer, epoch, mnist, null);

            Path modelDir = Paths.get("build/mlp");
            Files.createDirectories(modelDir);
            model.setProperty("Epoch", String.valueOf(epoch));
            model.save(modelDir, "mlp");
            System.out.println(model);
        }

    }
}
