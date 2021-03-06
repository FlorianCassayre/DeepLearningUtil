package me.cassayre.florian.dpu.util.mnist;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public final class MNISTTrainingImage
{
    public static final int SIZE = 28, LENGTH = SIZE * SIZE;

    private final byte[] data;

    public MNISTTrainingImage(byte[] data)
    {
        this.data = data; // Warning, no copy
    }

    public Volume imageToVolume()
    {
        final Volume volume = new Volume(new Dimensions(SIZE, SIZE, 1));

        int i = 0;
        for(int y = 0; y < SIZE; y++)
        {
            for(int x = 0; x < SIZE; x++)
            {
                volume.set(x, y, 0, (data[i] & 0xff) / 255.0);

                i++;
            }
        }

        return volume;
    }

    public int pixelAt(int i)
    {
        if(!(i >= 0 && i < LENGTH))
            throw new IndexOutOfBoundsException();

        return data[i];
    }

    public int pixelAt(int x, int y)
    {
        if(!isInBound(x, y))
            throw new IndexOutOfBoundsException();

        return data[x + y * SIZE] & 0xff;
    }

    private boolean isInBound(int x, int y)
    {
        return x >= 0 && y >= 0 && x < SIZE && y < SIZE;
    }

    public MNISTTrainingImage translated(int rx, int ry)
    {
        final byte[] array = new byte[LENGTH];

        for(int y = 0; y < SIZE; y++)
        {
            for(int x = 0; x < SIZE; x++)
            {
                final byte b = isInBound(x + rx, y + ry) ? data[x + rx + (y + ry) * SIZE] : 0;

                array[x + y * SIZE] = b;
            }
        }

        return new MNISTTrainingImage(array);
    }

    @Override
    public String toString()
    {
        final StringBuilder builder = new StringBuilder();
        for(int y = 0; y < SIZE; y++)
        {
            for(int x = 0; x < SIZE; x++)
            {
                builder.append(pixelAt(x, y) != 0 ? "#" : " ");
            }

            if(y < SIZE - 1)
                builder.append("\n");
        }

        return builder.toString();
    }
}
