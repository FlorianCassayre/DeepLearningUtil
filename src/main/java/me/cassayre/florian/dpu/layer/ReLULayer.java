package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class ReLULayer extends Layer
{
    public ReLULayer(Dimensions dimensions)
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
        volume.fillValues(i -> Math.max(input.get(i), 0.0));
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients(i -> input.get(i) > 0.0 ? volume.getGradient(i) : 0.0);
    }
}
