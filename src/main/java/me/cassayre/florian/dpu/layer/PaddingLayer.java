package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class PaddingLayer extends Layer
{
    private final Dimensions inputDimensions;
    private final int padX, padY;

    public PaddingLayer(Dimensions inputDimensions, int preX, int subX, int preY, int subY)
    {
        super(new Dimensions(inputDimensions.getWidth() + preX + subX, inputDimensions.getHeight() + preY + subY, inputDimensions.getDepth()));

        if(inputDimensions.getWidth() + preX + subX < 0 || -subX >= inputDimensions.getWidth() || -preX >= inputDimensions.getWidth() || inputDimensions.getHeight() + preY + subY < 0 || -subY >= inputDimensions.getHeight() || -preY >= inputDimensions.getHeight())
            throw new IllegalArgumentException("Output volume must be strictly positive");

        this.inputDimensions = inputDimensions;
        this.padX = Math.max(preX, 0);
        this.padY = Math.max(preY, 0);
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return inputDimensions;
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        volume.fillValues((x, y, z) ->
        {
            if(isInBounds(x - padX, y - padY))
                return input.get(x - padX, y - padY, z);
            return 0.0;
        });
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        // Not working properly
        volume.fillGradients((x, y, z) ->
        {
            if(isInBounds(x - padX, y - padY))
                return input.getGradient(x - padX, y - padY, z);
            return 0.0;
        });
    }

    private boolean isInBounds(int x, int y)
    {
        return x >= 0 && y >= 0 && x < inputDimensions.getWidth() && y < inputDimensions.getHeight();
    }
}
