package me.cassayre.florian.dpu.util;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

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
        volume.fillValues(i -> (Math.random() - 0.5));
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
}
