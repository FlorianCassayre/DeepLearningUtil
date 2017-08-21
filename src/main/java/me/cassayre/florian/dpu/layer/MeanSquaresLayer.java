package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Volume;

public class MeanSquaresLayer extends OutputLayer
{
    public MeanSquaresLayer(Dimensions dimensions)
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
        volume.fillValues(input::get);
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
                    final double v = volume.get(x, y, z) - expected.get(x, y, z);

                    volume.setGradient(x, y, z, v);

                    sum += v * v;
                }
            }
        }

        loss = sum;
    }
}
