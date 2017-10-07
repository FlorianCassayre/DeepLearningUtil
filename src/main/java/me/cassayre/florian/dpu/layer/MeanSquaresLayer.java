package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

import java.util.function.Function;

public class MeanSquaresLayer extends OutputLayer
{
    public MeanSquaresLayer(Dimensions dimensions)
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
        input.fillGradients((Function<Integer, Double>) volume::getGradient);
    }

    @Override
    public void backwardPropagationExpected(Volume expected)
    {
        loss = 0.0;

        volume.foreach(i ->
        {
            final double v = volume.get(i) - expected.get(i);

            volume.setGradient(i, v);

            loss += v * v;
        });
    }
}
