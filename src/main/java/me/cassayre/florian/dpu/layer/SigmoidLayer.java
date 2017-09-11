package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class SigmoidLayer extends Layer
{
    public SigmoidLayer(Dimensions dimensions)
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
        volume.fillValues((x, y, z) -> sigmoid(input.get(x, y, z)));
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients((x, y, z) ->
        {
            final double v = volume.get(x, y, z);
            return v * (1 - v) * volume.getGradient(x, y, z);
        });
    }

    private double sigmoid(double x)
    {
        return 1.0 / (1 + Math.exp(-x));
    }
}
