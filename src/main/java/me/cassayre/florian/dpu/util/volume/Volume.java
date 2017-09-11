package me.cassayre.florian.dpu.util.volume;

import me.cassayre.florian.dpu.util.TriConsumer;
import me.cassayre.florian.dpu.util.TriFunction;

import java.util.function.Consumer;
import java.util.function.Function;

public final class Volume
{
    private final Dimensions dimensions;
    private final double[] values;
    private final double[] gradient;

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

    public Dimensions getDimensions()
    {
        return dimensions;
    }

    public int getWidth()
    {
        return dimensions.getWidth();
    }

    public int getHeight()
    {
        return dimensions.getHeight();
    }

    public int getDepth()
    {
        return dimensions.getDepth();
    }

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
}
