package me.cassayre.florian.dpu.layer;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

public class ConvolutionLayer extends Layer
{
    protected final Volume[] filters;
    protected final Volume biases;

    public ConvolutionLayer(Dimensions imageDimensions, Volume[] filters, Volume biases) // filter: (width, height, previous_depth)[next_depth]
    {
        super(new Dimensions(imageDimensions.getWidth(), imageDimensions.getHeight(), filters.length));

        if(filters[0].getWidth() % 2 == 0 || filters[0].getHeight() % 2 == 0)
            throw new IllegalArgumentException("Filter dimensions must be odd");

        this.filters = filters;
        this.biases = biases; // One dimensional
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return new Dimensions(volume.getWidth(), volume.getHeight(), filters[0].getDepth());
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        for(int i = 0; i < volume.getDepth(); i++)
        {
            final Volume filter = this.filters[i];

            for(int x = 0; x < volume.getWidth(); x++)
            {
                for(int y = 0; y < volume.getHeight(); y++)
                {
                    double sum = biases.get(0, 0, i);

                    final int rx = (filters[0].getWidth() - 1) >> 1, ry = (filters[0].getHeight() - 1) >> 1;

                    for(int x1 = -rx; x1 <= rx; x1++)
                    {
                        for(int y1 = -ry; y1 <= ry; y1++)
                        {
                            final int xf = x + x1, yf = y + y1;

                            if(isInBounds(xf, yf))
                            {
                                for(int j = 0; j < filter.getDepth(); j++)
                                {
                                    sum += input.get(xf, yf, j) * filter.get(x1 + rx, y1 + rx, j);
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

        for(int i = 0; i < volume.getDepth(); i++)
        {
            final Volume filter = filters[i];

            for(int x = 0; x < volume.getWidth(); x++)
            {
                for(int y = 0; y < volume.getHeight(); y++)
                {
                    final double chain = volume.getGradient(x, y, i);

                    final int rx = (filters[0].getWidth() - 1) >> 1, ry = (filters[0].getHeight() - 1) >> 1;

                    for(int x1 = -rx; x1 <= rx; x1++)
                    {
                        for(int y1 = -ry; y1 <= ry; y1++)
                        {
                            final int xf = x + x1, yf = y + y1;

                            if(isInBounds(xf, yf))
                            {
                                for(int j = 0; j < filter.getDepth(); j++)
                                {
                                    filter.addGradient(x1 + rx, y1 + ry, j, input.get(xf, yf, j) * chain);
                                    input.addGradient(xf, yf, j, filter.get(x1 + rx, y1 + ry, j) * chain);
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

    @Override
    public JsonObject export()
    {
        final JsonObject object = new JsonObject();
        final JsonArray array = new JsonArray();

        for(Volume filter : filters)
        {
            final JsonObject f = new JsonObject();
            final JsonObject w = new JsonObject();

            int i = 0;
            for(int y = 0; y < filter.getHeight(); y++)
            {
                for(int x = 0; x < filter.getWidth(); x++)
                {
                    for(int z = 0; z < filter.getDepth(); z++)
                    {
                        w.add(i + "", new JsonPrimitive(filter.get(x, y, z)));
                        i++;
                    }
                }
            }

            f.add("w", w);
            array.add(f);
        }

        object.add("filters", array);

        final JsonObject b = new JsonObject();
        final JsonObject w = new JsonObject();

        for(int i = 0; i < biases.getDepth(); i++)
        {
            w.add(i + "", new JsonPrimitive(biases.get(0, 0, i)));
        }

        b.add("w", w);
        object.add("biases", b);

        return object;
    }
}
