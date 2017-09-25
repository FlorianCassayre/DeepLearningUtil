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
    private final List<LayerParameters> parameters;

    public Network(InputLayer inputLayer, List<Layer> hiddenLayers, OutputLayer outputLayer, List<LayerParameters> parameters)
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

        if(parameters.size() != layers.size())
            throw new IllegalArgumentException("Parameters don't match one by one to their corresponding layer");

        this.parameters = Collections.unmodifiableList(new ArrayList<>(parameters));
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

    public List<LayerParameters> getParameters()
    {
        return parameters;
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

        private final List<LayerParameters> parameters = new ArrayList<>();
        private boolean isTrainable;

        private boolean isBuilt = false;

        public Builder(Dimensions inputDimensions)
        {
            inputLayer = new InputLayer(inputDimensions);
            previous = inputLayer;

            setDefaultParameters();
        }

        public Builder hookFullyConnected(Volume[] weights, Volume biases, Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            hookLayer(new FullyConnectedLayer(weights, biases));

            hookActivationFunction(functionType);

            return this;
        }

        public Builder hookFullyConnected(Dimensions dimensions, Layer.ActivationFunctionType functionType)
        {
            if(dimensions.getWidth() != 1 || dimensions.getHeight() != 1)
                throw new IllegalArgumentException();

            return hookFullyConnected(Utils.randomWeightsVolumeArray(previous.getOutputDimensions(), dimensions.getDepth()), Utils.randomWeightsVolume(dimensions), functionType);
        }

        public Builder hookConvolutionLayer(Volume[] filters, Volume biases, int convolutionStride, int convolutionPadding, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            hookLayer(new ConvolutionLayer(previous.getOutputDimensions(), filters, biases, convolutionStride, convolutionStride, convolutionPadding, convolutionPadding));

            hookActivationFunction(functionType);

            maxPool(poolingStride);

            return this;
        }

        public Builder hookConvolutionLayer(Dimensions filterDimensions, int convolutionStride, int convolutionPadding, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            return hookConvolutionLayer(Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), previous.getOutput().getDepth()), filterDimensions.getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()), convolutionStride, convolutionPadding, poolingStride, functionType);
        }

        public Builder hookConvolutionLayer(Dimensions filterDimensions, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            return hookConvolutionLayer(Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), previous.getOutput().getDepth()), filterDimensions.getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()), 1, filterDimensions.getWidth() >> 1, poolingStride, functionType);
        }

        public Builder hookDeconvolutionLayer(Volume[] filters, Volume biases, int convolutionStride, int convolutionPadding, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            checkBuilt();

            hookLayer(new DeconvolutionLayer(previous.getOutputDimensions(), filters, biases, convolutionStride, convolutionStride, convolutionPadding, convolutionPadding));

            hookActivationFunction(functionType);

            hookUpSample(poolingStride);

            return this;
        }

        public Builder hookDeconvolutionLayer(Dimensions filterDimensions, int convolutionStride, int convolutionPadding, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            return hookDeconvolutionLayer(Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), previous.getOutput().getDepth()), filterDimensions.getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()), convolutionStride, convolutionPadding, poolingStride, functionType);
        }

        public Builder hookDeconvolutionLayer(Dimensions filterDimensions, int poolingStride, Layer.ActivationFunctionType functionType)
        {
            return hookDeconvolutionLayer(Utils.randomWeightsVolumeArray(new Dimensions(filterDimensions.getWidth(), filterDimensions.getHeight(), previous.getOutput().getDepth()), filterDimensions.getDepth()), Utils.randomWeightsVolume(1, 1, filterDimensions.getDepth()), 1, filterDimensions.getWidth() >> 1, poolingStride, functionType);
        }

        public Builder maxPool(int poolingStride)
        {
            checkBuilt();

            if(poolingStride > 1)
            {
                hookLayer(new MaxPoolingLayer(previous.getOutputDimensions(), poolingStride));
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

                hookLayer(function);
            }

            return this;
        }

        public Builder reshape(Dimensions newDimensions)
        {
            checkBuilt();

            hookLayer(new ReshapeLayer(previous.getOutputDimensions(), newDimensions));

            return this;
        }

        public Builder pad(int preX, int subX, int preY, int subY)
        {
            checkBuilt();

            hookLayer(new PaddingLayer(previous.getOutputDimensions(), preX, subX, preY, subY));

            return this;
        }

        public Builder hookUpSample(int stride)
        {
            checkBuilt();

            hookLayer(new UpSampleLayer(previous.getOutputDimensions(), stride));

            return this;
        }

        public Builder hookBilinearResample(Dimensions newDimensions)
        {
            checkBuilt();

            hookLayer(new BilinearResample(previous.getOutputDimensions(), newDimensions));

            return this;
        }

        public Builder setTrainable(boolean trainable)
        {
            isTrainable = trainable;

            return this;
        }

        public void hookLayer(Layer layer)
        {
            hiddenLayers.add(layer);
            previous = layer;

            pushParameters();
        }

        private void pushParameters()
        {
            parameters.add(new LayerParameters(isTrainable));
            setDefaultParameters();
        }

        private void setDefaultParameters()
        {
            isTrainable = true;
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

            pushParameters();

            pushParameters(); // Output layer

            isBuilt = true;

            return new Network(inputLayer, hiddenLayers, outputLayer, parameters);
        }

        private void checkBuilt()
        {
            if(isBuilt)
                throw new IllegalStateException("Network already built");
        }
    }
}
