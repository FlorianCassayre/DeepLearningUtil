package me.cassayre.florian.dpu.network;

public final class LayerParameters
{
    private final boolean trainable;

    public LayerParameters(boolean trainable)
    {
        this.trainable = trainable;
    }

    public boolean isTrainable()
    {
        return trainable;
    }
}
