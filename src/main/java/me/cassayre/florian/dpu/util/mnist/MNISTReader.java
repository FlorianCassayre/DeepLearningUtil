package me.cassayre.florian.dpu.util.mnist;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public interface MNISTReader
{
    static int IMAGES_MAGIC_VALUE = 0x00000803;
    static int LABELS_MAGIC_VALUE = 0x00000801;

    static String TRAINING_IMAGES_FILE_NAME = "train-images.idx3-ubyte";
    static String TRAINING_LABELS_FILE_NAME = "train-labels.idx1-ubyte";

    static String TEST_IMAGES_FILE_NAME = "t10k-images.idx3-ubyte";
    static String TEST_LABELS_FILE_NAME = "t10k-labels.idx1-ubyte";

    static List<TrainingImage> readImages(String fileName) throws IOException
    {
        final DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(fileName)));

        if(in.readInt() != IMAGES_MAGIC_VALUE)
            throw new IllegalStateException();

        final int count = in.readInt();
        final int rows = in.readInt(), columns = in.readInt();

        if(rows != TrainingImage.SIZE || columns != TrainingImage.SIZE)
            throw new IllegalStateException();

        final List<TrainingImage> list = new ArrayList<>(count);

        for(int i = 0; i < count; i++)
        {
            final byte[] array = new byte[TrainingImage.SIZE * TrainingImage.SIZE];

            in.read(array);

            list.add(new TrainingImage(array));
        }

        return list;
    }

    static List<Integer> readLabels(String fileName) throws IOException
    {
        final DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(fileName)));

        if(in.readInt() != LABELS_MAGIC_VALUE)
            throw new IllegalStateException();

        final int count = in.readInt();

        final List<Integer> list = new ArrayList<>(count);

        for(int i = 0; i < count; i++)
        {
            list.add(in.read() & 0xff);
        }

        return list;
    }

    static List<TrainingImage> readTrainingImages() throws IOException
    {
        return readImages(TRAINING_IMAGES_FILE_NAME);
    }

    static List<Integer> readTrainingLabels() throws IOException
    {
        return readLabels(TRAINING_LABELS_FILE_NAME);
    }

    static List<TrainingImage> readTestImages() throws IOException
    {
        return readImages(TEST_IMAGES_FILE_NAME);
    }

    static List<Integer> readTestLabels() throws IOException
    {
        return readLabels(TEST_LABELS_FILE_NAME);
    }
}
