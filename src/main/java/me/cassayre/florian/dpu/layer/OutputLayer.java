package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Volume;

public abstract class OutputLayer extends Layer
{
    protected double loss;

    public OutputLayer(Dimensions dimensions)
    {
        super(dimensions);
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return volume.getDimensions();
    }

    public abstract void backwardPropagationExpected(Volume expected);

    public double getLoss()
    {
        return loss;
    }
}
