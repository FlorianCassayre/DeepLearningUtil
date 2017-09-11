package me.cassayre.florian.dpu.util.cifar;

import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Volume;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class CIFAR10TrainingImage
{
    public static final int SIZE = 32, SIZE_SQ = SIZE * SIZE, LENGTH = 3 * SIZE_SQ;

    public static List<String> LABELS = Collections.unmodifiableList(Arrays.asList("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"));

    private final byte[] data;
    private final int label;

    public CIFAR10TrainingImage(int label, byte[] data)
    {
        this.data = data; // Warning, no copy
        this.label = label;
    }

    public Volume imageToVolume()
    {
        final Volume volume = new Volume(new Dimensions(SIZE, SIZE, 3));

        int i = 0;
        for(int j = 0; j < 3; j++)
        {
            for(int y = 0; y < SIZE; y++)
            {
                for(int x = 0; x < SIZE; x++)
                {
                    volume.set(x, y, j, data[i] / 255.0);

                    i++;
                }
            }
        }

        return volume;
    }

    public Volume labelToVolume()
    {
        final Volume volume = new Volume(new Dimensions(1, 1, LABELS.size()));

        volume.set(0, 0, label, 1.0);

        return volume;
    }

    public int getLabel()
    {
        return label;
    }

    public String getLabelName()
    {
        return LABELS.get(label);
    }
}
