package me.cassayre.florian.dpu.network;

import me.cassayre.florian.dpu.layer.*;
import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.Utils;
import me.cassayre.florian.dpu.util.volume.Volume;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Network
{
    private final List<Layer> layers;

    public Network(InputLayer inputLayer, List<Layer> hiddenLayers, OutputLayer outputLayer)
    {
        final List<Layer> layers = new ArrayList<>(hiddenLayers.size() + 2);

        layers.add(inputLayer);
        layers.addAll(hiddenLayers);
        layers.add(outputLayer);

        Dimensions previous = inputLayer.getOutputDimensions();
        for(int i = 1; i < layers.size(); i++)
        {
            final Layer layer = layers.get(i);

            if(!layer.getInputDimensions().equals(previous))
                throw new IllegalArgumentException("Dimensions don't match: " + previous + " and " + layer.getInputDimensions() + " (previous output and next input)");

            previous = layer.getOutputDimensions();
        }

        this.layers = Collections.unmodifiableList(layers);
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

    public List<Layer> getLayers()
    {
        return layers;
    }

    @Deprecated
    public void clearGradients()
    {
        for(Layer layer : layers)
        {
            for(Volume volume : layer.getWeights())
            {
                volume.fillGradients((x, y, z) -> 0.0);
            }
        }
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

        public Builder fullyConnected(Volume[] weights, Volume biases, Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            layer(new FullyConnectedLayer(weights, biases));

            activationFunction(functionType);

            return this;
        }

        public Builder fullyConnected(Dimensions dimensions, Layer.ActivationFunctionType functionType)
        {
            if(dimensions.getWidth() != 1 || dimensions.getHeight() != 1)
                throw new IllegalArgumentException();

            return fullyConnected(Utils.randomWeightsVolumeArray(previous.getOutputDimensions(), dimensions.getDepth()), Utils.randomWeightsVolume(dimensions), functionType);
        }

        public Builder convolution(Volume[] filters, Volume biases, int convolutionStride, int convolutionPadding, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            layer(new ConvolutionLayer(previous.getOutputDimensions(), filters, biases, convolutionStride, convolutionStride, convolutionPadding, convolutionPadding));

            activationFunction(functionType);

            maxPool(poolingStride);

            return this;
        }

        public Builder convolution(Dimensions filterDimensions, int convolutionStride, int convolutionPadding, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            return convolution(Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), previous.getOutput().getDepth()), filterDimensions.getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()), convolutionStride, convolutionPadding, poolingStride, functionType);
        }

        public Builder convolution(Dimensions filterDimensions, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            return convolution(Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), previous.getOutput().getDepth()), filterDimensions.getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()), 1, filterDimensions.getWidth() >> 1, poolingStride, functionType);
        }

        public Builder deconvolution(Volume[] filters, Volume biases, int convolutionStride, int convolutionPadding, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            layer(new DeconvolutionLayer(previous.getOutputDimensions(), filters, biases, convolutionStride, convolutionStride, convolutionPadding, convolutionPadding));

            activationFunction(functionType);

            if(poolingStride != 1)
                upSample(poolingStride);

            return this;
        }

        public Builder deconvolution(Dimensions filterDimensions, int convolutionStride, int convolutionPadding, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            return deconvolution(Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), previous.getOutput().getDepth()), filterDimensions.getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()), convolutionStride, convolutionPadding, poolingStride, functionType);
        }

        public Builder deconvolution(Dimensions filterDimensions, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            return deconvolution(Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), previous.getOutput().getDepth()), filterDimensions.getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()), 1, filterDimensions.getWidth() >> 1, poolingStride, functionType);
        }

        public Builder maxPool(int poolingStride)
        {
            checkBuilt();

            if(poolingStride > 1)
            {
                layer(new MaxPoolingLayer(previous.getOutputDimensions(), poolingStride));
            }

            return this;
        }

        public Builder activationFunction(Layer.ActivationFunctionType functionType)
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

                layer(function);
            }

            return this;
        }

        public Builder reshape(Dimensions newDimensions)
        {
            checkBuilt();

            layer(new ReshapeLayer(previous.getOutputDimensions(), newDimensions));

            return this;
        }

        public Builder pad(int preX, int subX, int preY, int subY)
        {
            checkBuilt();

            layer(new PaddingLayer(previous.getOutputDimensions(), preX, subX, preY, subY));

            return this;
        }

        public Builder upSample(int stride)
        {
            checkBuilt();

            layer(new UpSampleLayer(previous.getOutputDimensions(), stride));

            return this;
        }

        public Builder bilinearResample(Dimensions newDimensions)
        {
            checkBuilt();

            layer(new BilinearResample(previous.getOutputDimensions(), newDimensions));

            return this;
        }

        public void layer(Layer layer)
        {
            hiddenLayers.add(layer);
            previous = layer;
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
