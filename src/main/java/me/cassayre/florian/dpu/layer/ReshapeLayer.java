package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class ReshapeLayer extends Layer
{
    private final Dimensions inputDimensions;

    public ReshapeLayer(Dimensions inputDimensions, Dimensions outputDimensions)
    {
        super(outputDimensions);

        this.inputDimensions = inputDimensions;
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return inputDimensions;
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        int i = 0;
        for(int z = 0; z < volume.getDepth(); z++)
        {
            for(int y = 0; y < volume.getHeight(); y++)
            {
                for(int x = 0; x < volume.getWidth(); x++)
                {
                    volume.set(x, y, z, input.get(i));
                    i++;
                }
            }
        }
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        int i = 0;
        for(int z = 0; z < volume.getDepth(); z++)
        {
            for(int y = 0; y < volume.getHeight(); y++)
            {
                for(int x = 0; x < volume.getWidth(); x++)
                {
                    input.setGradient(i, volume.getGradient(x, y, z));
                    i++;
                }
            }
        }
    }
}
