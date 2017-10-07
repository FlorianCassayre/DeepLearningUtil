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
        for(Layer layer : network.getLayers())
        {
            final Volume[] weights = layer.getWeights();

            if(!layer.isTrainable())
                continue;

            for(final Volume weightVolume : weights)
            {
                weightVolume.fillValues(k -> weightVolume.get(k) - learningRate * weightVolume.getGradient(k) / batchSize);
            }
        }
    }
}
