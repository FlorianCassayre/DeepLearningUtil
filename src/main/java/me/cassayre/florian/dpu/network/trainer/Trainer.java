package me.cassayre.florian.dpu.network.trainer;

import me.cassayre.florian.dpu.layer.Layer;
import me.cassayre.florian.dpu.network.Network;
import me.cassayre.florian.dpu.util.Volume;

public abstract class Trainer
{
    protected final Network network;
    protected final int batchSize;

    protected int seen = 0;
    protected double loss;

    public Trainer(Network network, int batchSize)
    {
        this.network = network;
        this.batchSize = batchSize;
    }

    public Trainer(Network network)
    {
        this(network, 1);
    }

    public double getLoss()
    {
        return loss;
    }

    public int getSeen()
    {
        return seen;
    }

    public void train(Volume input, Volume expectedOutput)
    {
        network.forwardPropagation(input);

        network.backwardPropagation(expectedOutput);

        loss = network.getLoss();

        seen++;

        if(seen % batchSize == 0)
        {
            updateWeights();

            for(Layer layer : network.layers)
                for(final Volume weightVolume : layer.getWeights())
                    weightVolume.fillGradients((x, y, z) -> 0.0);
        }
    }

    protected abstract void updateWeights();
}
