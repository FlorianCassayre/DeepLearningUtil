package me.cassayre.florian.dpu.util.volume;

import java.util.Objects;

public final class Dimensions
{
    private final int width, height, depth;

    public Dimensions(int width, int height, int depth)
    {
        if(width < 1 || height < 1 || depth < 1)
            throw new IllegalArgumentException("Dimensions must be strictly positive");

        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    public Dimensions(int width, int height)
    {
        this(width, height, 1);
    }

    public Dimensions(int depth)
    {
        this(1, 1, depth);
    }

    public int getWidth()
    {
        return width;
    }

    public int getHeight()
    {
        return height;
    }

    public int getDepth()
    {
        return depth;
    }

    public int getSize()
    {
        return width * height * depth;
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(width, height, depth);
    }

    @Override
    public boolean equals(Object o)
    {
        if(!(o instanceof Dimensions))
            return false;
        final Dimensions that = (Dimensions) o;

        return width == that.width && height == that.height && depth == that.depth;
    }

    @Override
    public String toString()
    {
        return "[" + width + ", " + height + ", " + depth + "]";
    }
}
