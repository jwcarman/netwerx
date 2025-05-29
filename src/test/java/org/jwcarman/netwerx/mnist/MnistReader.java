package org.jwcarman.netwerx.mnist;

import org.apache.commons.io.function.IOFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

public class MnistReader {

// ------------------------------ FIELDS ------------------------------

    public static final int IMAGES_MAGIC_NUMBER = 2051;
    public static final int LABELS_MAGIC_NUMBER = 2049;
    public static final String TRAINING_LABELS = "/dataset/mnist/train-labels-idx1-ubyte.gz";
    public static final String TEST_IMAGES = "/dataset/mnist/t10k-images-idx3-ubyte.gz";
    public static final String TEST_LABELS = "/dataset/mnist/t10k-labels-idx1-ubyte.gz";
    public static final String TRAINING_IMAGES = "/dataset/mnist/train-images-idx3-ubyte.gz";

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> M readTrainingImages(MatrixFactory<M> factory) {
        return readTrainingImages(Integer.MAX_VALUE, factory);
    }

    public static <M extends Matrix<M>> M readTrainingImages(int nImages, MatrixFactory<M> factory) {
        return readImages(nImages, factory, TRAINING_IMAGES);
    }

    private static <M extends Matrix<M>> M readImages(int nImages, MatrixFactory<M> factory, String resourcePath) {
        return readFromResource(resourcePath, din -> {
            var magic = din.readInt();
            if (magic != IMAGES_MAGIC_NUMBER) {
                throw new RuntimeException("Invalid MNIST image file format. Expected magic number 2051, got " + magic);
            }
            var imageCount = Math.min(din.readInt(), nImages);
            var rows = din.readInt();
            var cols = din.readInt();
            var imageSize = rows * cols;
            double[][] data = new double[imageSize][imageCount];
            for (int col = 0; col < imageCount; col++) {
                for (int row = 0; row < imageSize; row++) {
                    data[row][col] = din.readUnsignedByte() / 255.0;
                }
            }
            return factory.from(data);
        });
    }

    private static <T> T readFromResource(String resourcePath, IOFunction<DataInputStream, T> reader) {
        try (var in = resourceStream(resourcePath);
             var zin = new GZIPInputStream(in);
             var din = new DataInputStream(zin)) {
            return reader.apply(din);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read from resource: " + resourcePath, e);
        }
    }

    private static InputStream resourceStream(String resourcePath) {
        InputStream in = MnistReader.class.getResourceAsStream(resourcePath);
        if (in == null) {
            throw new IllegalArgumentException("Resource not found: " + resourcePath);
        }
        return in;
    }

    public static int[] readTrainingLabels() {
        return readTrainingLabels(Integer.MAX_VALUE);
    }

    public static int[] readTrainingLabels(int nLabels) {
        return readLabels(nLabels, TRAINING_LABELS);
    }

    private static int[] readLabels(int nLabels, String resourcePath) {
        return readFromResource(resourcePath, din -> {
            var magic = din.readInt();
            if (magic != LABELS_MAGIC_NUMBER) {
                throw new RuntimeException("Invalid MNIST labels file format. Expected magic number 2049, got " + magic);
            }
            var labelCount = Math.min(din.readInt(), nLabels);
            var data = new int[labelCount];
            for (int i = 0; i < labelCount; i++) {
                data[i] = din.readUnsignedByte();
            }
            return data;
        });
    }

    public static <M extends Matrix<M>> M readTestImages(MatrixFactory<M> factory) {
        return readTestImages(Integer.MAX_VALUE, factory);
    }

    public static <M extends Matrix<M>> M readTestImages(int nImages, MatrixFactory<M> factory) {
        return readImages(nImages, factory, TEST_IMAGES);
    }

    public static int[] readTestLabels() {
        return readTestLabels(Integer.MAX_VALUE);
    }

    public static int[] readTestLabels(int nLabels) {
        return readLabels(nLabels, TEST_LABELS);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private MnistReader() {
        // Prevent instantiation
    }

}
