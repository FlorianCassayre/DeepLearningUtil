package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Volume;

public class SoftmaxLayer extends OutputLayer
{
    private static final double LN_2 = Math.log(2);

    public SoftmaxLayer(Dimensions dimensions)
    {
        super(dimensions);
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return volume.getDimensions();
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        double max = Double.NEGATIVE_INFINITY;
        for(int x = 0; x < input.getWidth(); x++)
        {
            for(int y = 0; y < input.getHeight(); y++)
            {
                for(int z = 0; z < input.getDepth(); z++)
                {
                    final double d = input.get(x, y, z);
                    max = Math.max(d, max);
                }
            }
        }

        double sum = 0.0;
        for(int x = 0; x < input.getWidth(); x++)
        {
            for(int y = 0; y < input.getHeight(); y++)
            {
                for(int z = 0; z < input.getDepth(); z++)
                {
                    final double d = input.get(x, y, z);
                    sum += Math.exp(d - max);
                }
            }
        }

        for(int x = 0; x < input.getWidth(); x++)
        {
            for(int y = 0; y < input.getHeight(); y++)
            {
                for(int z = 0; z < input.getDepth(); z++)
                {
                    volume.set(x, y, z, Math.exp(input.get(x, y, z) - max) / sum);
                }
            }
        }
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients(volume::getGradient);
    }

    @Override
    public void backwardPropagationExpected(Volume expected)
    {
        double sum = 0.0;

        for(int x = 0; x < expected.getWidth(); x++)
        {
            for(int y = 0; y < expected.getHeight(); y++)
            {
                for(int z = 0; z < expected.getDepth(); z++)
                {
                    volume.setGradient(x, y, z, (volume.get(x, y, z) - expected.get(x, y, z)));

                    final double exp = expected.get(x, y, z);
                    final double actual = volume.get(x, y, z);

                    final double l = exp * log2(actual);

                    sum -= l;
                }
            }
        }

        loss = sum;
    }

    private double log2(double x)
    {
        return Math.log(x) / LN_2;
    }
}
