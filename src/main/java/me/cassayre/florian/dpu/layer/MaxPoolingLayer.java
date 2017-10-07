package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class MaxPoolingLayer extends Layer
{
    protected final int stride;

    public MaxPoolingLayer(Dimensions inputDimensions, int stride)
    {
        super(new Dimensions(inputDimensions.getWidth() / stride, inputDimensions.getHeight() / stride, inputDimensions.getDepth()));

        if(inputDimensions.getWidth() % stride != 0 || inputDimensions.getHeight() % stride != 0)
            throw new IllegalArgumentException("Stride must divide the dimensions");

        this.stride = stride;
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return new Dimensions(volume.getWidth() * stride, volume.getHeight() * stride, volume.getDepth());
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        volume.foreach((x, y, z) ->
        {
            double max = Double.NEGATIVE_INFINITY;

            for(int y1 = 0; y1 < stride; y1++)
            {
                for(int x1 = 0; x1 < stride; x1++)
                {
                    max = Math.max(input.get(x * stride + x1, y * stride + y1, z), max);
                }
            }

            volume.set(x, y, z, max);
        });
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        volume.foreach((x, y, z) ->
        {
            final double chain = volume.getGradient(x, y, z);

            final double max = volume.get(x, y, z);

            for(int y1 = 0; y1 < stride; y1++)
            {
                for(int x1 = 0; x1 < stride; x1++)
                {
                    final int rx = x * stride + x1, ry = y * stride + y1;

                    input.setGradient(rx, ry, z, input.get(rx, ry, z) == max ? chain : 0.0);
                }
            }
        });
    }
}
