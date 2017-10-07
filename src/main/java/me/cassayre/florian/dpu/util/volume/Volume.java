package me.cassayre.florian.dpu.util.volume;

import me.cassayre.florian.dpu.util.TriConsumer;
import me.cassayre.florian.dpu.util.TriFunction;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Locale;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * A 3-dimensional tensor containing double precision float values as well as their gradients.
 * Once defined, the size of a volume cannot be modified.
 */
public final class Volume
{
    private final Dimensions dimensions;
    private final double[] values;
    private final double[] gradient;

    /**
     * Creates a new volume initialized with zeroes (values & gradients) having the specified {@link me.cassayre.florian.dpu.util.volume.Dimensions}.
     * @param dimensions the dimensions of the volume
     */
    public Volume(Dimensions dimensions)
    {
        this.dimensions = dimensions;

        this.values = new double[dimensions.getSize()];
        this.gradient = new double[dimensions.getSize()];
    }

    private Volume(Dimensions dimensions, double[] values, double[] gradient)
    {
        this.dimensions = dimensions;
        this.values = values;
        this.gradient = gradient;
    }

    /**
     * Returns the dimensions of the volume.
     * @return the dimensions
     */
    public Dimensions getDimensions()
    {
        return dimensions;
    }

    /**
     * @see me.cassayre.florian.dpu.util.volume.Dimensions#getWidth
     */
    public int getWidth()
    {
        return dimensions.getWidth();
    }

    /**
     * @see me.cassayre.florian.dpu.util.volume.Dimensions#getHeight
     */
    public int getHeight()
    {
        return dimensions.getHeight();
    }

    /**
     * @see me.cassayre.florian.dpu.util.volume.Dimensions#getDepth
     */
    public int getDepth()
    {
        return dimensions.getDepth();
    }

    /**
     * @see me.cassayre.florian.dpu.util.volume.Dimensions#getSize
     */
    public int getSize()
    {
        return dimensions.getSize();
    }

    private int getIndex(int x, int y, int z)
    {
        return z + (x + (y * dimensions.getWidth())) * dimensions.getDepth();
    }

    public double get(int x, int y, int z)
    {
        return values[getIndex(x, y, z)];
    }

    public double get(int i)
    {
        return values[i];
    }

    public void set(int x, int y, int z, double v)
    {
        values[getIndex(x, y, z)] = v;
    }

    public void set(int i, double v)
    {
        values[i] = v;
    }

    public void add(int x, int y, int z, double v)
    {
        values[getIndex(x, y, z)] += v;
    }

    public void add(int i, double v)
    {
        values[i] += v;
    }

    public double getGradient(int x, int y, int z)
    {
        return gradient[getIndex(x, y, z)];
    }

    public double getGradient(int i)
    {
        return gradient[i];
    }

    public void setGradient(int x, int y, int z, double v)
    {
        gradient[getIndex(x, y, z)] = v;
    }

    public void setGradient(int i, double v)
    {
        gradient[i] = v;
    }

    public void addGradient(int x, int y, int z, double v)
    {
        gradient[getIndex(x, y, z)] += v;
    }

    public void addGradient(int i, double v)
    {
        gradient[i] += v;
    }

    public void foreach(TriConsumer<Integer, Integer, Integer> consumer)
    {
        for(int x = 0; x < getWidth(); x++)
        {
            for(int y = 0; y < getHeight(); y++)
            {
                for(int z = 0; z < getDepth(); z++)
                {
                    consumer.accept(x, y, z);
                }
            }
        }
    }

    public void foreach(Consumer<Integer> consumer)
    {
        for(int i = 0; i < dimensions.getSize(); i++)
        {
            consumer.accept(i);
        }
    }

    public void fillValues(TriFunction<Integer, Integer, Integer, Double> function)
    {
        foreach((x, y, z) -> set(x, y, z, function.apply(x, y, z)));
    }

    public void fillValues(Function<Integer, Double> function)
    {
        foreach(i -> set(i, function.apply(i)));
    }

    public void fillGradients(TriFunction<Integer, Integer, Integer, Double> function)
    {
        foreach((x, y, z) -> setGradient(x, y, z, function.apply(x, y, z)));
    }

    public void fillGradients(Function<Integer, Double> function)
    {
        foreach(i -> setGradient(i, function.apply(i)));
    }

    public void fillValuesRelative(TriFunction<Integer, Integer, Integer, Double> function)
    {
        foreach((x, y, z) -> add(x, y, z, function.apply(x, y, z)));
    }

    public void fillValuesRelative(Function<Integer, Double> function)
    {
        foreach(i -> add(i, function.apply(i)));
    }

    public void fillGradientsRelative(TriFunction<Integer, Integer, Integer, Double> function)
    {
        foreach((x, y, z) -> addGradient(x, y, z, function.apply(x, y, z)));
    }

    public void fillGradientsRelative(Function<Integer, Double> function)
    {
        foreach(i -> addGradient(i, function.apply(i)));
    }

    @Override
    public Volume clone()
    {
        double[] values = new double[dimensions.getSize()];
        double[] gradient = new double[dimensions.getSize()];

        foreach(i ->
        {
            values[i] = this.values[i];
            gradient[i] = this.gradient[i];
        });

        return new Volume(dimensions, values, gradient);
    }

    @Override
    public String toString()
    {
        final String open = "[", close = "]", comma = ",";
        final NumberFormat nf = NumberFormat.getNumberInstance(Locale.US);
        final DecimalFormat formatter = (DecimalFormat) nf;
        formatter.applyPattern("#0.000");

        final StringBuilder builder = new StringBuilder();
        builder.append(open);
        for(int z = 0; z < getDepth(); z++)
        {
            builder.append(open);
            for(int y = 0; y < getHeight(); y++)
            {
                builder.append(open);
                for(int x = 0; x < getWidth(); x++)
                {
                    builder.append(formatter.format(get(x, y, z)));
                    if(x < getWidth() - 1)
                        builder.append(comma);
                }
                builder.append(close);
                if(y < getHeight() - 1)
                    builder.append(comma);
            }
            builder.append(close);
            if(z < getDepth() - 1)
                builder.append(comma);
        }
        builder.append(close);

        return builder.toString();
    }
}
