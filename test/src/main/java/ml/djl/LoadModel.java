package ml.djl;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;

/**
 *
 *
 * @author Jiawei Mao
 * @version 1.0.0
 * @since 26 Nov 2025, 7:26 PM
 */
public class LoadModel {
    static void main() {
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .build();
    }
}
