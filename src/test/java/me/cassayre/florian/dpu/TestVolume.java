package me.cassayre.florian.dpu;

import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class TestVolume
{
    @Test(expected = NullPointerException.class)
    public void testThrowForNullDimensions()
    {
        new Volume(null);
    }

    @Test
    public void testSetAndGetValue()
    {
        final Volume volume = new Volume(new Dimensions(1));
        assertEquals(0.0, volume.get(0), 0.0);
        volume.set(0, 1.0);
        assertEquals(1.0, volume.get(0), 0.0);
    }

    @Test
    public void testCloneAndImmutability()
    {
        final Volume volume1 = new Volume(new Dimensions(1));
        volume1.set(0, 1.0);
        final Volume volume2 = volume1.clone();
        assertEquals(1.0, volume2.get(0), 0.0);
        volume1.set(0, 1.0);
        assertEquals(1.0, volume2.get(0), 0.0);
    }

    @Test
    public void testIterationOrder()
    {
        final Random random = new Random(0);
        final Volume volume = new Volume(new Dimensions(2, 3, 4));

        int i = 0;
        for(int y = 0; y < volume.getHeight(); y++)
        {
            for(int x = 0; x < volume.getWidth(); x++)
            {
                for(int z = 0; z < volume.getDepth(); z++)
                {
                    final double d = random.nextDouble();
                    volume.set(x, y, z, d);
                    assertEquals(d, volume.get(i), 0.0);
                    i++;
                }
            }
        }
    }
}
