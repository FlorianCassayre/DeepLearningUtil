package me.cassayre.florian.dpu.util.cifar;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public interface CIFARReader
{
    static int COUNT_PER_BATCH = 10000;

    static String FILE_PREFIX = "data_batch_", FILE_SUFFIX = ".bin";
    static String FILE_TEST = "test_batch.bin";

    static List<CIFAR10TrainingImage> readBatch(String fileName) throws IOException
    {
        final DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(fileName)));

        final List<CIFAR10TrainingImage> list = new ArrayList<>(COUNT_PER_BATCH);

        for(int i = 0; i < COUNT_PER_BATCH; i++)
        {
            final int label = in.read();

            final byte[] array = new byte[CIFAR10TrainingImage.LENGTH];

            in.read(array);

            list.add(new CIFAR10TrainingImage(label, array));
        }

        return list;
    }
    
    static List<CIFAR10TrainingImage> readAllBatches() throws IOException
    {
        List<CIFAR10TrainingImage> list = new ArrayList<>();

        for(int i = 1; i <= 5; i++)
        {
            list.addAll(readBatch(FILE_PREFIX + i + FILE_SUFFIX));
        }

        return list;
    }

    static List<CIFAR10TrainingImage> readTestBatch() throws IOException
    {
        return readBatch(FILE_TEST);
    }
}
