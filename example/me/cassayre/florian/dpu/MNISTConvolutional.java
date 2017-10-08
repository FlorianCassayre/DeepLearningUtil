package me.cassayre.florian.dpu;

import me.cassayre.florian.dpu.layer.Layer;
import me.cassayre.florian.dpu.network.architecture.FeedForwardNetwork;
import me.cassayre.florian.dpu.network.Network;
import me.cassayre.florian.dpu.network.trainer.AdadeltaTrainer;
import me.cassayre.florian.dpu.network.trainer.Trainer;
import me.cassayre.florian.dpu.util.mnist.MNISTReader;
import me.cassayre.florian.dpu.util.mnist.MNISTTrainingImage;
import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

import java.util.List;

public class MNISTConvolutional
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

        // Network described here: http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
        final Network network = new FeedForwardNetwork.Builder(new Dimensions(24, 24, 1))
                .convolution(new Dimensions(5, 5, 8), 2, Layer.ActivationFunctionType.RELU)
                .convolution(new Dimensions(5, 5, 16), 3, Layer.ActivationFunctionType.RELU)
                .fullyConnected(new Dimensions(10), Layer.ActivationFunctionType.LINEAR)
                .build(Layer.OutputFunctionType.SOFTMAX);

        // Mini-batch size: 1
        final Trainer trainer = new AdadeltaTrainer(network, 0.95, 1E-6);

        final List<MNISTTrainingImage> images = MNISTReader.readTrainingImages();
        final List<Integer> labels = MNISTReader.readTrainingLabels();

        final List<MNISTTrainingImage> testImages = MNISTReader.readTestImages();
        final List<Integer> testLabels = MNISTReader.readTestLabels();

        final int sampling = 100; // Prints the state every 100 images

        double sum = 0.0;
        int correct = 0;

        for(int i = 0; i < images.size(); i++)
        {
            final MNISTTrainingImage image = images.get(i);
            final int label = labels.get(i);

            final Volume input = imageTo24Volume(image); // Input

            final Volume expected = new Volume(new Dimensions(10)); // Expected output
            expected.set(0, 0, label, 1.0);

            trainer.train(input, expected);

            final Volume actual = network.getOutput(); // Actual output
            final double loss = trainer.getLoss();

            final int max = getActivation(actual);

            sum += loss;

            if(max == label)
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

        for(int i = 0; i < testImages.size(); i++)
        {
            final MNISTTrainingImage testImage = testImages.get(i);
            final int testLabel = testLabels.get(i);

            network.forwardPropagation(imageTo24Volume(testImage));

            if(getActivation(network.getOutput()) == testLabel)
                correct++;
        }

        System.out.println("Test accuracy: " + (100.0 * correct / testImages.size()) + "%");
    }

    private static Volume imageTo24Volume(MNISTTrainingImage image)
    {
        final Volume volume = new Volume(new Dimensions(24, 24, 1));

        for(int x = 0; x < volume.getWidth(); x++)
        {
            for(int y = 0; y < volume.getHeight(); y++)
            {
                volume.set(x, y, 0, image.pixelAt(x + 2, y + 2) / 255.0);
            }
        }

        return volume;
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
