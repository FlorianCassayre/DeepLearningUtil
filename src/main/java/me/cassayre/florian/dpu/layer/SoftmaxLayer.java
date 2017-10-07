package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

import java.util.function.Function;

public class SoftmaxLayer extends OutputLayer
{
    private double max, sum;

    private static final double LN_2 = Math.log(2);

    public SoftmaxLayer(Dimensions dimensions)
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
        max = Double.NEGATIVE_INFINITY;
        input.foreach(i ->
        {
            final double d = input.get(i);
            max = Math.max(d, max);
        });

        sum = 0.0;
        input.foreach(i ->
        {
            final double d = input.get(i);
            sum += Math.exp(d - max);
        });

        input.foreach(i ->
        {
            volume.set(i, Math.exp(input.get(i) - max) / sum);
        });
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

        expected.foreach(i ->
        {
            volume.setGradient(i, (volume.get(i) - expected.get(i)));

            final double exp = expected.get(i);
            final double actual = volume.get(i);

            final double l = exp * log2(actual);

            loss -= l;
        });
    }

    private double log2(double x)
    {
        return Math.log(x) / LN_2;
    }
}
