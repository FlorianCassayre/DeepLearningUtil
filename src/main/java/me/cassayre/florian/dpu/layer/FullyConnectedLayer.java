package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class FullyConnectedLayer extends Layer
{
    protected final Volume[] weights;
    protected final Volume biases;

    public FullyConnectedLayer(Volume[] weights, Volume biases)
    {
        super(new Dimensions(weights.length));

        this.weights = weights;
        this.biases = biases;
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return weights[0].getDimensions();
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        for(int i = 0; i < weights.length; i++)
        {
            final int j = i;
            final Volume multipliers = weights[i];
            final double bias = biases.get(0, 0, i);

            volume.set(0, 0, i, bias);

            input.foreach(k -> volume.add(0, 0, j, input.get(k) * multipliers.get(k)));
        }
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients(k -> 0.0);

        for(int i = 0; i < weights.length; i++)
        {
            final Volume multipliers = weights[i];
            final double chain = volume.getGradient(0, 0, i);

            input.foreach(k ->
            {
                input.addGradient(k, multipliers.get(k) * chain);
                multipliers.addGradient(k, input.get(k) * chain);
            });

            biases.addGradient(0, 0, i, chain);
        }
    }

    @Override
    public Volume[] getWeights()
    {
        final Volume[] array = new Volume[weights.length + 1];
        System.arraycopy(weights, 0, array, 0, weights.length);
        array[array.length - 1] = biases;

        return array;
    }
}
