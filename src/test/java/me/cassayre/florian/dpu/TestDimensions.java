package me.cassayre.florian.dpu;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class TestDimensions
{
    @Test
    public void testCorrectDimensions()
    {
        final Dimensions dimensions = new Dimensions(2, 3, 4);
        assertEquals(2, dimensions.getWidth());
        assertEquals(3, dimensions.getHeight());
        assertEquals(4, dimensions.getDepth());
        assertEquals(24, dimensions.getSize());
    }

    @Test
    public void testCorrectDimensionsOtherConstructors()
    {
        final Dimensions dimensions1 = new Dimensions(2);
        final Dimensions dimensions2 = new Dimensions(2, 3);
        assertEquals(1, dimensions1.getWidth());
        assertEquals(1, dimensions1.getHeight());
        assertEquals(2, dimensions1.getDepth());
        assertEquals(2, dimensions1.getSize());
        assertEquals(2, dimensions2.getWidth());
        assertEquals(3, dimensions2.getHeight());
        assertEquals(1, dimensions2.getDepth());
        assertEquals(6, dimensions2.getSize());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testThrowForIllegalDimensions1()
    {
        new Dimensions(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testThrowForIllegalDimensions2()
    {
        new Dimensions(-1, 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testThrowForIllegalDimensions3()
    {
        new Dimensions(1, -10, 1);
    }

    @Test
    public void testHashCodeAndEquals()
    {
        final Dimensions dimensions1 = new Dimensions(2, 3);
        final Dimensions dimensions2 = new Dimensions(2, 3, 4);
        final Dimensions dimensions3 = new Dimensions(2, 3, 4);
        assertNotEquals(dimensions1, dimensions2);
        assertEquals(dimensions2.hashCode(), dimensions3.hashCode());
        assertEquals(dimensions2, dimensions3);
    }
}
