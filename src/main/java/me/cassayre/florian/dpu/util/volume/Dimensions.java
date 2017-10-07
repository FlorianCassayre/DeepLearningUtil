package me.cassayre.florian.dpu.util.volume;

import java.util.Objects;

/**
 * An immutable class representing the dimensions of a {@link me.cassayre.florian.dpu.util.volume.Volume}.
 */
public final class Dimensions
{
    private final int width, height, depth;

    /**
     * Creates a new space of dimensionality <code>(width, height, depth)</code>.
     * @param width the width of the volume
     * @param height the height of the volume
     * @param depth the depth of the volume
     */
    public Dimensions(int width, int height, int depth)
    {
        if(width < 1 || height < 1 || depth < 1)
            throw new IllegalArgumentException("Dimensions must be strictly positive");

        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    /**
     * Creates a new space of dimensionality <code>(width, height, 1)</code>.
     * @param width the width of the volume
     * @param height the height of the volume
     */
    public Dimensions(int width, int height)
    {
        this(width, height, 1);
    }

    /**
     * Creates a new space of dimensionality <code>(1, 1, depth)</code>.
     * @param depth the depth of the volume
     */
    public Dimensions(int depth)
    {
        this(1, 1, depth);
    }

    /**
     * Creates a new space of dimensionality <code>(1, 1, 1)</code>.
     */
    public Dimensions()
    {
        this(1, 1, 1);
    }

    /**
     * The number of values on the x-axis.
     * @return the width of the volume
     */
    public int getWidth()
    {
        return width;
    }

    /**
     * The number of values on the y-axis.
     * @return the height of the volume
     */
    public int getHeight()
    {
        return height;
    }

    /**
     * The number of values on the z-axis.
     * @return the depth of the volume
     */
    public int getDepth()
    {
        return depth;
    }

    /**
     * The number of values in the volume, namely <code>width*height*depth</code>.
     * @return the size of the volume
     */
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
