package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Volume;

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
        volume.fillValues((x, y, z) -> Math.max(input.get(x, y, z), 0.0));
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients((x, y, z) -> input.get(x, y, z) > 0.0 ? volume.getGradient(x, y, z) : 0.0);
    }
}
