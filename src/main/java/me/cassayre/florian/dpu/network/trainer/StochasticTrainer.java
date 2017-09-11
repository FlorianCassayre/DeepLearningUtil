package me.cassayre.florian.dpu.network.trainer;

import me.cassayre.florian.dpu.layer.Layer;
import me.cassayre.florian.dpu.network.Network;
import me.cassayre.florian.dpu.util.volume.Volume;

public class StochasticTrainer extends Trainer
{
    private final double learningRate;

    public StochasticTrainer(Network network, int batchSize, double learningRate)
    {
        super(network, batchSize);

        this.learningRate = learningRate;
    }

    public StochasticTrainer(Network network, double learningRate)
    {
        super(network);

        this.learningRate = learningRate;
    }

    @Override
    protected void updateWeights()
    {
        for(Layer layer : network.layers)
        {
            final Volume[] weights = layer.getWeights();

            for(final Volume weightVolume : weights)
            {
                weightVolume.fillValues((x, y, z) -> weightVolume.get(x, y, z) - learningRate * weightVolume.getGradient(x, y, z) / batchSize);
            }
        }
    }
}
