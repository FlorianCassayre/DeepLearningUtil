package me.cassayre.florian.dpu.network;

import me.cassayre.florian.dpu.layer.*;
import me.cassayre.florian.dpu.util.Dimensions;
import me.cassayre.florian.dpu.util.Utils;
import me.cassayre.florian.dpu.util.Volume;

import java.util.ArrayList;
import java.util.List;

public class Network
{
    public final List<Layer> layers;

    public Network(InputLayer inputLayer, List<Layer> hiddenLayers, OutputLayer outputLayer)
    {
        this.layers = new ArrayList<>(hiddenLayers.size() + 2);

        this.layers.add(inputLayer);
        this.layers.addAll(hiddenLayers);
        this.layers.add(outputLayer);

        Dimensions previous = inputLayer.getOutputDimensions();
        for(int i = 1; i < layers.size(); i++)
        {
            final Layer layer = layers.get(i);

            if(!layer.getInputDimensions().equals(previous))
                throw new IllegalArgumentException("Dimensions don't match: " + previous + " and " + layer.getInputDimensions() + " (previous output and next input)");

            previous = layer.getOutputDimensions();
        }
    }

    public void forwardPropagation(Volume input)
    {
        Volume previous = input;
        for(Layer layer : layers)
        {
            layer.forwardPropagation(previous);
            previous = layer.getOutput();
        }
    }

    public void backwardPropagation(Volume expectedOutput)
    {
        ((OutputLayer) layers.get(layers.size() - 1)).backwardPropagationExpected(expectedOutput); // Output layer

        for(int i = layers.size() - 1; i >= 1; i--)
        {
            final Layer current = layers.get(i);
            final Volume previous = layers.get(i - 1).getOutput();

            current.backwardPropagation(previous);
        }
    }

    public Volume getOutput()
    {
        return layers.get(layers.size() - 1).getOutput();
    }

    public double getLoss()
    {
        return ((OutputLayer) layers.get(layers.size() - 1)).getLoss();
    }

    public static class Builder
    {
        private InputLayer inputLayer;
        private List<Layer> hiddenLayers = new ArrayList<>();
        private OutputLayer outputLayer;

        private Layer previous;

        private boolean isBuilt = false;

        public Builder(Dimensions inputDimensions)
        {
            inputLayer = new InputLayer(inputDimensions);
            previous = inputLayer;
        }

        public Builder hookFullyConnected(Dimensions dimensions, Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            if(dimensions.getWidth() != 1 || dimensions.getHeight() != 1)
                throw new IllegalArgumentException();

            final FullyConnectedLayer layer = new FullyConnectedLayer(Utils.randomWeightsVolumeArray(previous.getOutputDimensions(), dimensions.getDepth()), Utils.randomWeightsVolume(dimensions));
            hiddenLayers.add(layer);

            previous = layer;

            hookActivationFunction(functionType);

            return this;
        }

        public Builder hookConvolutionLayer(Dimensions filterDimensions, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            final ConvolutionLayer layer = new ConvolutionLayer(previous.getOutputDimensions(), Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), filterDimensions.getDepth()), previous.getOutput().getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()));

            hiddenLayers.add(layer);

            previous = layer;

            hookActivationFunction(functionType);

            if(poolingStride > 1)
            {
                final MaxPoolingLayer poolingLayer = new MaxPoolingLayer(previous.getOutputDimensions(), poolingStride);
                hiddenLayers.add(poolingLayer);

                previous = poolingLayer;
            }

            return this;
        }

        public Builder hookActivationFunction(Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            if(functionType != Layer.ActivationFunctionType.LINEAR)
            {
                Layer function;
                if(functionType == Layer.ActivationFunctionType.RELU)
                {
                    function = new ReLULayer(previous.getOutputDimensions());
                }
                else if(functionType == Layer.ActivationFunctionType.SIGMOID)
                {
                    function = new SigmoidLayer(previous.getOutputDimensions());
                }
                else if(functionType == Layer.ActivationFunctionType.TANH)
                {
                    function = new TanhLayer(previous.getOutputDimensions());
                }
                else
                {
                    throw new UnsupportedOperationException();
                }

                hiddenLayers.add(function);
                previous = function;
            }

            return this;
        }

        public Network build(Layer.OutputFunctionType outputFunctionType)
        {
            checkBuilt();

            if(outputFunctionType == Layer.OutputFunctionType.SOFTMAX)
            {
                outputLayer = new SoftmaxLayer(previous.getOutputDimensions());
            }
            else if(outputFunctionType == Layer.OutputFunctionType.MEAN_SQUARES)
            {
                outputLayer = new MeanSquaresLayer(previous.getOutputDimensions());
            }
            else
            {
                throw new UnsupportedOperationException();
            }

            isBuilt = true;

            return new Network(inputLayer, hiddenLayers, outputLayer);
        }

        private void checkBuilt()
        {
            if(isBuilt)
                throw new IllegalStateException("Network already built");
        }
    }
}
