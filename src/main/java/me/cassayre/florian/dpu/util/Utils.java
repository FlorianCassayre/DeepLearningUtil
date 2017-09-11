package me.cassayre.florian.dpu.util;

import me.cassayre.florian.dpu.util.mnist.MNISTTrainingImage;

public final class Utils
{
    private Utils()
    {}

    public static boolean areSameDimensions(Volume volume1, Volume volume2)
    {
        return volume1.getWidth() == volume2.getWidth() && volume1.getHeight() == volume2.getHeight() && volume1.getDepth() == volume2.getDepth();
    }

    public static Volume randomWeightsVolume(Dimensions dimensions)
    {
        final Volume volume = new Volume(dimensions);
        volume.fillValues((x, y, z) -> (Math.random() - 0.5) * 2);
        return volume;
    }

    public static Volume randomWeightsVolume(int width, int height, int depth)
    {
        return randomWeightsVolume(new Dimensions(width, height, depth));
    }

    public static Volume[] randomWeightsVolumeArray(Dimensions dimensions, int n)
    {
        final Volume[] volumes = new Volume[n];
        for(int i = 0; i < n; i++)
        {
            volumes[i] = randomWeightsVolume(dimensions);
        }
        return volumes;
    }

    public static Volume[] randomWeightsVolumeArray(int width, int height, int depth, int n)
    {
        return randomWeightsVolumeArray(new Dimensions(width, height, depth), n);
    }

    @Deprecated
    public static Volume imageTo24Volume(MNISTTrainingImage image)
    {
        final Volume volume = new Volume(new Dimensions(24, 24, 1));

        for(int x = 0; x < volume.getWidth(); x++)
        {
            for(int y = 0; y < volume.getHeight(); y++)
            {
                volume.set(x, y, 0, image.pixelAt(x + 2, y + 2) / 255.0);
            }
        }

        return volume;
    }

    @Deprecated
    public static int getMNISTActivation(Volume output)
    {
        int k = -1;

        for(int i = 0; i < 10; i++)
        {
            if(k == -1 || output.get(0, 0, i) > output.get(0, 0, k))
            {
                k = i;
            }
        }

        return k;
    }
}
