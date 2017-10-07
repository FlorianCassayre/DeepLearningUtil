package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class TanhLayer extends Layer
{
    public TanhLayer(Dimensions dimensions)
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
        volume.fillValues(i -> Math.tanh(input.get(i)));
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients(i -> 1 - square(volume.get(i)));
    }

    private double square(double x)
    {
        return x * x;
    }
}
