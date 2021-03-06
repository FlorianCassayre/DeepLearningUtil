package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class DeconvolutionLayer extends Layer
{
    private final Volume[] filters;
    private final Volume biases;
    private final Dimensions inputDimensions;
    private final int strideX, strideY;
    private final int paddingX, paddingY;

    public DeconvolutionLayer(Dimensions imageDimensions, Volume[] filters, Volume biases, int strideX, int strideY, int paddingX, int paddingY) // filter: (width, height, previous_depth)[next_depth]
    {
        super(new Dimensions(strideX * (imageDimensions.getWidth() - 1) + 1 + 2 * paddingX, strideY * (imageDimensions.getHeight() - 1) + 1 + 2 * paddingY, filters.length));

        if(filters.length != biases.getDepth())
            throw new IllegalArgumentException("Biases must match the weights");

        for(int i = 1; i < filters.length; i++)
            if(!filters[i].getDimensions().equals(filters[0].getDimensions()))
                throw new IllegalArgumentException("Filter weights must have the same dimensions");

        if(filters[0].getWidth() % 2 == 0 || filters[0].getHeight() % 2 == 0)
            throw new IllegalArgumentException("Filter dimensions must be odd");

        if(strideX <= 0 || strideY <= 0)
            throw new IllegalArgumentException("Strides must be strictly positive");

        if(paddingX < 0 || paddingY < 0)
            throw new IllegalArgumentException("Padding must be positive"); // Let's not accept this case, even though it can work (as long as the resulting volume is big enough)

        this.inputDimensions = imageDimensions;

        this.strideX = strideX;
        this.strideY = strideY;
        this.paddingX = paddingX;
        this.paddingY = paddingY;

        this.filters = filters;
        this.biases = biases; // One dimensional
    }

    public DeconvolutionLayer(Dimensions imageDimensions, Volume[] filters, Volume biases)
    {
        this(imageDimensions, filters, biases, 1, 1, filters[0].getWidth() >> 1, filters[0].getHeight() >> 1);
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return inputDimensions;
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        volume.fillValues((i) -> 0.0);

        final int rx = (filters[0].getWidth() - 1) >> 1, ry = (filters[0].getHeight() - 1) >> 1;

        for(int i = 0; i < volume.getDepth(); i++)
        {
            final Volume filter = this.filters[i];

            for(int y = 0; y < input.getHeight(); y++)
            {
                final int cy = y * strideY + paddingY; // Center output
                for(int x = 0; x < input.getWidth(); x++)
                {
                    final int cx = x * strideX + paddingX;

                    for(int y1 = -ry; y1 <= ry; y1++)
                    {
                        final int yf = cy + y1;
                        if(!isYInBounds(yf))
                            continue;
                        for(int x1 = -rx; x1 <= rx; x1++)
                        {
                            final int xf = cx + x1;
                            if(!isXInBounds(xf))
                                continue;
                            for(int j = 0; j < filter.getDepth(); j++)
                            {
                                volume.add(xf, yf, i, input.get(x, y, j) * filter.get(rx + x1, ry + y1, j));
                            }
                        }
                    }
                }
            }

            for(int y = 0; y < volume.getHeight(); y++)
            {
                for(int x = 0; x < volume.getWidth(); x++)
                {
                    volume.add(x, y, i, biases.get(i));
                }
            }
        }
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients(i -> 0.0);

        final int rx = (filters[0].getWidth() - 1) >> 1, ry = (filters[0].getHeight() - 1) >> 1;

        for(int i = 0; i < volume.getDepth(); i++)
        {
            final Volume filter = this.filters[i];

            for(int y = 0; y < input.getHeight(); y++)
            {
                final int cy = y * strideY + paddingY; // Center output
                for(int x = 0; x < input.getWidth(); x++)
                {
                    final int cx = x * strideX + paddingX;

                    for(int y1 = -ry; y1 <= ry; y1++)
                    {
                        final int yf = cy + y1;
                        if(!isYInBounds(yf))
                            continue;
                        for(int x1 = -rx; x1 <= rx; x1++)
                        {
                            final int xf = cx + x1;
                            if(!isXInBounds(xf))
                                continue;

                            for(int j = 0; j < filter.getDepth(); j++)
                            {
                                filter.addGradient(rx + x1, ry + y1, j, volume.getGradient(xf, yf, i) * input.get(x, y, j));
                                input.addGradient(x, y, j, volume.getGradient(xf, yf, i) * filter.get(rx + x1, ry + y1, j));
                            }
                        }
                    }
                }
            }

            for(int y = 0; y < volume.getHeight(); y++)
            {
                for(int x = 0; x < volume.getWidth(); x++)
                {
                    biases.addGradient(i, volume.getGradient(x, y, i));
                }
            }
        }
    }

    private boolean isXInBounds(int x)
    {
        return x >= 0 && x < volume.getWidth();
    }

    private boolean isYInBounds(int y)
    {
        return y >= 0 && y < volume.getHeight();
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
