package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class BilinearResample extends Layer
{
    private final Dimensions inputDimensions;

    public BilinearResample(Dimensions inputDimensions, Dimensions outputDimensions)
    {
        super(outputDimensions);

        if(inputDimensions.getDepth() != outputDimensions.getDepth())
            throw new IllegalArgumentException("Depths must be equal");

        this.inputDimensions = inputDimensions;
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return inputDimensions;
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        for(int z = 0; z < volume.getDepth(); z++)
        {
            for(int y = 0; y < volume.getHeight(); y++)
            {
                final double yCurrentScaled = (double) y / (volume.getHeight() - 1);
                final int ya = (int) Math.floor(yCurrentScaled * (input.getHeight() - 1)), yb = Math.min(ya + 1, input.getHeight() - 1);
                final double yaScaled = (double) ya / (input.getHeight() - 1);
                final double dy = 1.0 / (input.getHeight() - 1);

                final double vert = (yCurrentScaled - yaScaled) / dy;

                for(int x = 0; x < volume.getWidth(); x++)
                {
                    final double xCurrentScaled = (double) x / (volume.getWidth() - 1);
                    final int xa = (int) Math.floor(xCurrentScaled * (input.getWidth() - 1)), xb = Math.min(xa + 1, input.getWidth() - 1);
                    final double xaScaled = (double) xa / (input.getWidth() - 1);
                    final double dx = 1.0 / (input.getWidth() - 1);

                    final double hor = (xCurrentScaled - xaScaled) / dx;

                    final double vaa = input.get(xa, ya, z), vba = input.get(xb, ya, z);
                    final double v1 = hor * (vba - vaa) + vaa;

                    final double vab = input.get(xa, yb, z), vbb = input.get(xb, yb, z);
                    final double v2 = hor * (vbb - vab) + vab;

                    final double vf = vert * (v2 - v1) + v1;

                    volume.set(x, y, z, vf);
                }
            }
        }
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients(i -> 0.0);

        for(int z = 0; z < input.getDepth(); z++)
        {
            for(int y = 0; y < input.getHeight(); y++)
            {
                final double yCurrentScaled = (double) y / (input.getHeight() - 1);
                final int ya = (int) Math.floor(yCurrentScaled * (volume.getHeight() - 1)), yb = Math.min(ya + 1, volume.getHeight() - 1);
                final double yaScaled = (double) ya / (volume.getHeight() - 1);
                final double dy = 1.0 / (volume.getHeight() - 1);

                final double vert = (yCurrentScaled - yaScaled) / dy;

                for(int x = 0; x < input.getWidth(); x++)
                {
                    final double xCurrentScaled = (double) x / (input.getWidth() - 1);
                    final int xa = (int) Math.floor(xCurrentScaled * (volume.getWidth() - 1)), xb = Math.min(xa + 1, volume.getWidth() - 1);
                    final double xaScaled = (double) xa / (volume.getWidth() - 1);
                    final double dx = 1.0 / (volume.getWidth() - 1);

                    final double hor = (xCurrentScaled - xaScaled) / dx;

                    final double vaa = volume.getGradient(xa, ya, z), vba = volume.getGradient(xb, ya, z);
                    final double v1 = hor * (vba - vaa) + vaa;

                    final double vab = volume.getGradient(xa, yb, z), vbb = volume.getGradient(xb, yb, z);
                    final double v2 = hor * (vbb - vab) + vab;

                    final double vf = vert * (v2 - v1) + v1;

                    input.addGradient(x, y, z, vf);
                }
            }
        }
    }
}
