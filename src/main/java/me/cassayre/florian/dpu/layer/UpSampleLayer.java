package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class UpSampleLayer extends Layer
{
    private final Dimensions inputDimensions;
    private final int stride, strideSq;

    public UpSampleLayer(Dimensions inputDimensions, int stride)
    {
        super(new Dimensions(inputDimensions.getWidth() * stride, inputDimensions.getHeight() * stride, inputDimensions.getDepth()));

        this.inputDimensions = inputDimensions;
        this.stride = stride;
        this.strideSq = stride * stride;
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return inputDimensions;
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        input.foreach((x, y, z) ->
        {
            final double v = input.get(x, y, z);
            for(int y1 = 0; y1 < stride; y1++)
            {
                for(int x1 = 0; x1 < stride; x1++)
                {
                    volume.set(x * stride + x1, y * stride + y1, z, v);
                }
            }
        });
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.foreach((x, y, z) ->
        {
            double v = 0.0;
            for(int y1 = 0; y1 < stride; y1++)
            {
                for(int x1 = 0; x1 < stride; x1++)
                {
                    v += volume.getGradient(x * stride + x1, y * stride + y1, z);
                }
            }

            input.setGradient(x, y, z, v / strideSq);
        });
    }
}
