package me.cassayre.florian.dpu;

import me.cassayre.florian.dpu.layer.Layer;
import me.cassayre.florian.dpu.network.Network;
import me.cassayre.florian.dpu.network.trainer.AdadeltaTrainer;
import me.cassayre.florian.dpu.network.trainer.Trainer;
import me.cassayre.florian.dpu.util.cifar.CIFAR10TrainingImage;
import me.cassayre.florian.dpu.util.cifar.CIFARReader;
import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

import java.util.List;

public class CIFAR10Convolutional
{
    public static void main(String[] args) throws Exception
    {
        /*
        Requires the following files (located at the root):
        - train-images.idx3-ubyte
        - train-labels.idx1-ubyte
        - t10k-images.idx3-ubyte
        - t10k-labels.idx1-ubyte
         */

        // Network described here: http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
        final Network network = new Network.Builder(new Dimensions(32, 32, 3)) // 32x32x3
                .convolution(new Dimensions(5, 5, 16), 2, Layer.ActivationFunctionType.RELU) // 16x16x16
                .convolution(new Dimensions(5, 5, 20), 2, Layer.ActivationFunctionType.RELU) // 8x8x24
                .convolution(new Dimensions(5, 5, 20), 2, Layer.ActivationFunctionType.RELU) // 4x4x32
                .fullyConnected(new Dimensions(10), Layer.ActivationFunctionType.LINEAR)
                .build(Layer.OutputFunctionType.SOFTMAX);

        // Mini-batch size: 1
        // Learning rate: 0.0001
        final Trainer trainer = new AdadeltaTrainer(network, 1, 0.0001);

        final List<CIFAR10TrainingImage> trainingSet = CIFARReader.readAllBatches();
        final List<CIFAR10TrainingImage> testingSet = CIFARReader.readTestBatch();

        final int sampling = 100; // Prints the state every 100 images

        double sum = 0.0;
        int correct = 0;

        for(int i = 0; i < trainingSet.size(); i++)
        {
            final CIFAR10TrainingImage trainingData = trainingSet.get(i);

            final Volume input = trainingData.imageToVolume(); // Input
            final Volume expected = trainingData.labelToVolume(); // Expected output

            trainer.train(input, expected);

            final Volume actual = network.getOutput(); // Actual output
            final double loss = trainer.getLoss();

            final int max = getActivation(actual);

            sum += loss;

            if(max == trainingData.getLabel())
                correct++;

            if(i != 0 && i % sampling == 0)
            {
                final double average = sum / sampling;

                System.out.println("Seen: " + i + "\tLoss: " + average + "\tAccuracy: " + ((double) correct / sampling));

                sum = 0.0;
                correct = 0;
            }

        }

        correct = 0;

        for(final CIFAR10TrainingImage testingData : testingSet)
        {
            network.forwardPropagation(testingData.imageToVolume());

            if(getActivation(network.getOutput()) == testingData.getLabel())
                correct++;
        }

        System.out.println("Test accuracy: " + (100.0 * correct / testingSet.size()) + "%");
    }

    private static int getActivation(Volume output)
    {
        int k = -1;

        for(int i = 0; i < 10; i++)
        {
            if(k == -1 || output.get(0, 0, i) > output.get(0, 0, k))
            {
                k = i;
            }
        }

        return k;
    }
}
