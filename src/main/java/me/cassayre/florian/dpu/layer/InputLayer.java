package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

import java.util.function.Function;

public class InputLayer extends Layer
{
    public InputLayer(Dimensions dimensions)
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
        volume.fillValues((Function<Integer, Double>) input::get);
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        // Empty
    }
}
