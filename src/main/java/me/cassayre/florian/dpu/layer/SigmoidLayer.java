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
        volume.fillValues(i -> sigmoid(input.get(i)));
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients(i ->
        {
            final double v = volume.get(i);
            return v * (1 - v) * volume.getGradient(i);
        });
    }

    private double sigmoid(double x)
    {
        return 1.0 / (1 + Math.exp(-x));
    }
}
