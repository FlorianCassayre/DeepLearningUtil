package me.cassayre.florian.dpu.layer;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class ConvolutionLayer extends Layer
{
    private final Volume[] filters;
    private final Volume biases;
    private final Dimensions inputDimensions;
    private final int strideX, strideY;
    private final int paddingX, paddingY;

    public ConvolutionLayer(Dimensions imageDimensions, Volume[] filters, Volume biases, int strideX, int strideY, int paddingX, int paddingY) // filter: (width, height, previous_depth)[next_depth]
    {
        super(new Dimensions((imageDimensions.getWidth() - filters[0].getWidth() + 2 * paddingX) / strideX + 1, (imageDimensions.getHeight() - filters[0].getHeight() + 2 * paddingY) / strideY + 1, filters.length));

        for(int i = 1; i < filters.length; i++)
            if(!filters[i].getDimensions().equals(filters[0].getDimensions()))
                throw new IllegalArgumentException("Filter weights must have the same dimensions");

        if(filters[0].getWidth() % 2 == 0 || filters[0].getHeight() % 2 == 0)
            throw new IllegalArgumentException("Filter dimensions must be odd");

        if(strideX <= 0 || strideY <= 0)
            throw new IllegalArgumentException("Strides must be strictly positive");

        if(paddingX < 0 || paddingY < 0)
            throw new IllegalArgumentException("Padding must be positive"); // Let's not accept this case, even though it can work (as long as the resulting volume is big enough)

        if((imageDimensions.getWidth() - filters[0].getWidth() + 2 * paddingX) % strideX != 0 || (imageDimensions.getHeight() - filters[0].getHeight() + 2 * paddingY) % strideY != 0)
            throw new IllegalArgumentException("Stride does not divide");

        this.inputDimensions = imageDimensions;

        this.strideX = strideX;
        this.strideY = strideY;
        this.paddingX = paddingX;
        this.paddingY = paddingY;

        this.filters = filters;
        this.biases = biases; // One dimensional
    }

    public ConvolutionLayer(Dimensions imageDimensions, Volume[] filters, Volume biases)
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
        final int rx = (filters[0].getWidth() - 1) >> 1, ry = (filters[0].getHeight() - 1) >> 1;

        for(int i = 0; i < volume.getDepth(); i++)
        {
            final Volume filter = this.filters[i];
            final int sx = (filter.getWidth() >> 1) - paddingX, sy = (filter.getHeight() >> 1) - paddingY;

            for(int y = 0; y < volume.getHeight(); y++)
            {
                for(int x = 0; x < volume.getWidth(); x++)
                {
                    double sum = biases.get(0, 0, i);

                    for(int y1 = -ry; y1 <= ry; y1++)
                    {
                        final int yf = y * strideY + y1 + sy;
                        if(!isYBounds(yf))
                            continue;
                        for(int x1 = -rx; x1 <= rx; x1++)
                        {
                            final int xf = x * strideX + x1 + sx;
                            if(!isXBounds(xf))
                                continue;
                            for(int j = 0; j < filter.getDepth(); j++)
                            {
                                sum += input.get(xf, yf, j) * filter.get(x1 + rx, y1 + rx, j);
                            }
                        }
                    }

                    volume.set(x, y, i, sum);
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
            final Volume filter = filters[i];
            final int sx = (filter.getWidth() >> 1) - paddingX, sy = (filter.getHeight() >> 1) - paddingY;

            for(int y = 0; y < volume.getHeight(); y++)
            {
                for(int x = 0; x < volume.getWidth(); x++)
                {
                    final double chain = volume.getGradient(x, y, i);

                    for(int y1 = -ry; y1 <= ry; y1++)
                    {
                        final int yf = y * strideY + y1 + sy;
                        if(!isYBounds(yf))
                            continue;
                        for(int x1 = -rx; x1 <= rx; x1++)
                        {
                            final int xf = x * strideX + x1 + sx;
                            if(!isXBounds(xf))
                                continue;
                            for(int j = 0; j < filter.getDepth(); j++)
                            {
                                filter.addGradient(x1 + rx, y1 + ry, j, input.get(xf, yf, j) * chain);
                                input.addGradient(xf, yf, j, filter.get(x1 + rx, y1 + ry, j) * chain);
                            }
                        }
                    }

                    biases.addGradient(0, 0, i, chain);
                }
            }
        }
    }

    private boolean isXBounds(int x)
    {
        return x >= 0 && x < inputDimensions.getWidth();
    }

    private boolean isYBounds(int y)
    {
        return y >= 0 && y < inputDimensions.getHeight();
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
