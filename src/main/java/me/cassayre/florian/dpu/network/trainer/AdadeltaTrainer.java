package me.cassayre.florian.dpu.network.trainer;

import me.cassayre.florian.dpu.network.LayerParameters;
import me.cassayre.florian.dpu.network.Network;
import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class AdadeltaTrainer extends Trainer
{
    private final double gamma, e;

    private Volume[][] gt, vt, xt;

    public AdadeltaTrainer(Network network, int batchSize, double gamma, double e)
    {
        super(network);

        this.gamma = gamma;
        this.e = e;

        gt = new Volume[network.getLayers().size()][];
        vt = new Volume[network.getLayers().size()][];
        xt = new Volume[network.getLayers().size()][];

        for(int i = 0; i < network.getLayers().size(); i++)
        {
            final Volume[] weights = network.getLayers().get(i).getWeights();

            gt[i] = new Volume[weights.length];
            vt[i] = new Volume[weights.length];
            xt[i] = new Volume[weights.length];

            for(int j = 0; j < weights.length; j++)
            {
                final Dimensions dimensions = weights[j].getDimensions();

                gt[i][j] = new Volume(dimensions);
                vt[i][j] = new Volume(dimensions);
                xt[i][j] = new Volume(dimensions);
            }
        }
    }

    public AdadeltaTrainer(Network network, double gamma, double e)
    {
        this(network, 1, gamma, e);
    }

    @Override
    protected void updateWeights()
    {
        for(int i = 0; i < network.getLayers().size(); i++)
        {
            final LayerParameters parameters = network.getParameters().get(i);
            final int i1 = i;
            final Volume[] weights = network.getLayers().get(i).getWeights();

            if(!parameters.isTrainable())
                continue;

            for(int j = 0; j < weights.length; j++)
            {
                final int j1 = j;
                final Volume weightVolume = weights[j];

                weightVolume.foreach((x, y, z) ->
                {
                    final double grad = weightVolume.getGradient(x, y, z) / batchSize;

                    final Volume gt1 = gt[i1][j1], vt1 = vt[i1][j1], xt1 = xt[i1][j1];

                    gt1.set(x, y, z, gamma * gt1.get(x, y, z) + (1 - gamma) * grad * grad);
                    vt1.set(x, y, z, -Math.sqrt(xt1.get(x, y, z) + e) * grad / Math.sqrt(gt1.get(x, y, z) + e));
                    xt1.set(x, y, z, gamma * xt1.get(x, y, z) + (1 - gamma) * vt1.get(x, y, z) * vt1.get(x, y, z));

                    weightVolume.add(x, y, z, vt1.get(x, y, z));
                });
            }
        }
    }
}
