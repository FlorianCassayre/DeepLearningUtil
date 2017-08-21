package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Volume;

public class ConvolutionLayer extends Layer
{
    protected final Volume[] filters;
    protected final Volume biases;

    public ConvolutionLayer(Dimensions imageDimensions, Volume[] filters, Volume biases)
    {
        super(new Dimensions(imageDimensions.getWidth(), imageDimensions.getHeight(), filters[0].getDepth()));

        if(filters[0].getWidth() % 2 == 0 || filters[0].getHeight() % 2 == 0)
            throw new IllegalArgumentException("Filter dimensions must be odd");

        this.filters = filters;
        this.biases = biases; // One dimensional
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return new Dimensions(volume.getWidth(), volume.getHeight(), filters.length);
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        for(int i = 0; i < volume.getDepth(); i++)
        {
            final double bias = biases.get(0, 0, i);

            for(int x = 0; x < volume.getWidth(); x++)
            {
                for(int y = 0; y < volume.getHeight(); y++)
                {
                    double sum = bias;

                    final int rx = (filters[0].getWidth() - 1) >> 1, ry = (filters[0].getHeight() - 1) >> 1;

                    for(int j = 0; j < filters.length; j++)
                    {
                        final Volume filter = filters[j];
                        for(int x1 = -rx; x1 <= rx; x1++)
                        {
                            for(int y1 = -ry; y1 <= ry; y1++)
                            {
                                final int xf = x + x1, yf = y + y1;
                                if(isInBounds(xf, yf))
                                {
                                    sum += input.get(xf, yf, j) * filter.get(x1 + rx, y1 + rx, i);
                                }
                            }
                        }
                    }

                    volume.set(x, y, i, sum);
                }
            }
        }
    }

    private boolean isInBounds(int x, int y)
    {
        return x >= 0 && y >= 0 && x < volume.getWidth() && y < volume.getHeight();
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients((x, y, z) -> 0.0);

        for(int i = 0; i < filters.length; i++)
        {
            final Volume filter = filters[i];

            for(int x = 0; x < volume.getWidth(); x++)
            {
                for(int y = 0; y < volume.getHeight(); y++)
                {
                    final double chain = volume.getGradient(x, y, i);

                    final int rx = (filters[0].getWidth() - 1) >> 1, ry = (filters[0].getHeight() - 1) >> 1;

                    for(int j = 0; j < volume.getDepth(); j++)
                    {
                        for(int x1 = -rx; x1 <= rx; x1++)
                        {
                            for(int y1 = -ry; y1 <= ry; y1++)
                            {
                                final int xf = x + x1, yf = y + y1;
                                if(isInBounds(xf, yf))
                                {
                                    filter.addGradient(x1 + rx, y1 + ry, j, input.get(xf, yf, i) * chain);
                                    input.addGradient(xf, yf, i, filter.get(x1 + rx, y1 + ry, j) * chain);
                                }
                            }
                        }
                    }

                    biases.addGradient(0, 0, i, chain);
                }
            }
        }
    }

    @Override
    public Volume[] getWeights()
    {
        final Volume[] array = new Volume[filters.length + 1];
        System.arraycopy(filters, 0, array, 0, filters.length);
        array[array.length - 1] = biases;

        return array;
    }
}
