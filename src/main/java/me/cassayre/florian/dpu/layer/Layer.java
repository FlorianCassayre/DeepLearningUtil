package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;

import java.io.Serializable;

import me.cassayre.florian.dpu.util.Utils;
import me.cassayre.florian.dpu.util.volume.Volume;

public abstract class Layer implements Serializable
{
    protected final Volume volume;
    private boolean isTrainable = true;

    public Layer(Dimensions dimensions)
    {
        this.volume = new Volume(dimensions);
    }

    public Volume getOutput()
    {
        return volume;
    }

    public abstract Dimensions getInputDimensions();

    public Dimensions getOutputDimensions()
    {
        return volume.getDimensions();
    }

    public abstract void forwardPropagation(Volume input);

    public abstract void backwardPropagation(Volume input);

    public Volume[] getWeights()
    {
        return new Volume[] {};
    }

    public boolean isTrainable()
    {
        return isTrainable;
    }

    public void setTrainable(boolean isTrainable)
    {
        this.isTrainable = isTrainable;
    }

    protected void checkSameDimensions(Volume input)
    {
        if(!Utils.areSameDimensions(input, volume))
            throw new IllegalArgumentException("Incompatible dimensions");
    }

    protected void checkOneDimensional(Volume input)
    {
        if(input.getWidth() != 1 || input.getHeight() != 1)
            throw new IllegalArgumentException("Input must be one dimensional");
    }

    public static enum ActivationFunctionType
    {
        LINEAR,
        RELU,
        SIGMOID,
        TANH;
    }

    public static enum OutputFunctionType
    {
        SOFTMAX,
        MEAN_SQUARES;
    }
}
