package me.cassayre.florian.dpu.util;

public final class Volume
{
    private final Dimensions dimensions;
    private final double[][][] values;
    private final double[][][] gradient;

    public Volume(Dimensions dimensions)
    {
        this.dimensions = dimensions;

        this.values = new double[dimensions.getWidth()][dimensions.getHeight()][dimensions.getDepth()];
        this.gradient = new double[dimensions.getWidth()][dimensions.getHeight()][dimensions.getDepth()];
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

    public double get(int x, int y, int z)
    {
        return values[x][y][z];
    }

    public void set(int x, int y, int z, double v)
    {
        values[x][y][z] = v;
    }

    public void add(int x, int y, int z, double v)
    {
        values[x][y][z] += v;
    }

    public double getGradient(int x, int y, int z)
    {
        return gradient[x][y][z];
    }

    public void setGradient(int x, int y, int z, double v)
    {
        gradient[x][y][z] = v;
    }

    public void addGradient(int x, int y, int z, double v)
    {
        gradient[x][y][z] += v;
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

    public void fillValues(TriFunction<Integer, Integer, Integer, Double> function)
    {
        foreach((x, y, z) -> set(x, y, z, function.apply(x, y, z)));
    }

    public void fillGradients(TriFunction<Integer, Integer, Integer, Double> function)
    {
        foreach((x, y, z) -> setGradient(x, y, z, function.apply(x, y, z)));
    }

    public void fillValuesRelative(TriFunction<Integer, Integer, Integer, Double> function)
    {
        foreach((x, y, z) -> add(x, y, z, function.apply(x, y, z)));
    }

    public void fillGradientsRelative(TriFunction<Integer, Integer, Integer, Double> function)
    {
        foreach((x, y, z) -> addGradient(x, y, z, function.apply(x, y, z)));
    }
}
