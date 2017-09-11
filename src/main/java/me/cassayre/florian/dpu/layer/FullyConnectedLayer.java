package me.cassayre.florian.dpu.layer;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Volume;

public class FullyConnectedLayer extends Layer
{
    protected final Volume[] weights;
    protected final Volume biases;

    public FullyConnectedLayer(Volume[] weights, Volume biases)
    {
        super(new Dimensions(weights.length));

        this.weights = weights;
        this.biases = biases;
    }

    @Override
    public Dimensions getInputDimensions()
    {
        return weights[0].getDimensions();
    }

    @Override
    public void forwardPropagation(Volume input)
    {
        for(int i = 0; i < weights.length; i++)
        {
            final int j = i;
            final Volume multipliers = weights[i];
            final double bias = biases.get(0, 0, i);

            volume.set(0, 0, i, bias);

            input.foreach((x, y, z) -> volume.add(0, 0, j, input.get(x, y, z) * multipliers.get(x, y, z)));
        }
    }

    @Override
    public void backwardPropagation(Volume input)
    {
        input.fillGradients((x, y, z) -> 0.0);

        for(int i = 0; i < weights.length; i++)
        {
            final Volume multipliers = weights[i];
            final double chain = volume.getGradient(0, 0, i);

            input.foreach((x, y, z) ->
            {
                input.addGradient(x, y, z, multipliers.get(x, y, z) * chain);
                multipliers.addGradient(x, y, z, input.get(x, y, z) * chain);
            });

            biases.addGradient(0, 0, i, chain);
        }
    }

    @Override
    public Volume[] getWeights()
    {
        final Volume[] array = new Volume[weights.length + 1];
        System.arraycopy(weights, 0, array, 0, weights.length);
        array[array.length - 1] = biases;

        return array;
    }

    @Override
    public JsonObject export()
    {
        final JsonObject object = new JsonObject();
        final JsonArray array = new JsonArray();


        for(int j = 0; j < weights.length; j++)
        {
            final JsonObject f = new JsonObject();
            final JsonObject w = new JsonObject();

            int i = 0;

            for(int y = 0; y < weights[0].getHeight(); y++)
            {
                for(int x = 0; x < weights[0].getWidth(); x++)
                {

                    for(int z = 0; z < weights[0].getDepth(); z++)
                    {
                        final Volume weight = weights[j];
                        w.add(i + "", new JsonPrimitive(weight.get(x, y, z)));
                        i++;
                    }
                }
            }

            f.add("w", w);
            array.add(f);
        }


        object.add("filters", array);

        final JsonObject b = new JsonObject();
        final JsonObject w2 = new JsonObject();

        for(int j = 0; j < biases.getDepth(); j++)
        {
            w2.add(j + "", new JsonPrimitive(biases.get(0, 0, j)));
        }

        b.add("w", w2);
        object.add("biases", b);

        return object;
    }
}
