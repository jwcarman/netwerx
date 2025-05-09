package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.matrix.Matrix;

public interface MultiClassifierTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Trains a multi-class classifier using the provided features and class labels.
     *
     * @param features the input features for training where each column represents a sample and each row represents a feature.
     * @param classes  the class labels for each sample, where each label is an integer representing the class index.
     * @return a trained MultiClassifier instance that can predict classes for new samples.
     */
    MultiClassifier<M> train(M features, int[] classes);

}
