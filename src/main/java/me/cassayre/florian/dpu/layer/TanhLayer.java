package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Volume;

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
        volume.fillValues((x, y, z) -> Math.tanh(input.get(x, y, z)));
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients((x, y, z) -> 1 - square(volume.get(x, y, z)));
    }

    private double square(double x)
    {
        return x * x;
    }
}
