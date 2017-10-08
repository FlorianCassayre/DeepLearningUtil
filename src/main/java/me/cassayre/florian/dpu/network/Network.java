package me.cassayre.florian.dpu.network;

import me.cassayre.florian.dpu.layer.Layer;
import me.cassayre.florian.dpu.util.volume.Volume;

import java.util.List;

public abstract class Network
{
    public abstract void forwardPropagation(Volume input);

    public abstract void backwardPropagation(Volume expectedOutput);

    public abstract Volume getOutput();

    public abstract double getLoss();

    public abstract List<Layer> getLayers();
}
