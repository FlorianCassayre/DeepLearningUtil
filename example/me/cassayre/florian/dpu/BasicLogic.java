package me.cassayre.florian.dpu;

import me.cassayre.florian.dpu.layer.Layer;
import me.cassayre.florian.dpu.network.Network;
import me.cassayre.florian.dpu.network.trainer.AdadeltaTrainer;
import me.cassayre.florian.dpu.network.trainer.Trainer;
import me.cassayre.florian.dpu.util.volume.Dimensions;
import me.cassayre.florian.dpu.util.volume.Volume;

import java.util.Random;
import java.util.function.BiFunction;

public class BasicLogic
{
    private static final int HIDDEN_UNITS = 10;
    // Too low: the network will reach a (global) optimum with a bad loss
    // Too high: the network will take more time to converge (and additionally, will be quadratically slower)

    private static final BiFunction<Boolean, Boolean, Boolean> OR = (a, b) -> a | b;
    private static final BiFunction<Boolean, Boolean, Boolean> AND = (a, b) -> a & b;
    private static final BiFunction<Boolean, Boolean, Boolean> XOR = (a, b) -> a ^ b;
    // ...

    public static void main(String[] args)
    {
        final Network network = new Network.Builder(new Dimensions(2))
                .fullyConnected(new Dimensions(HIDDEN_UNITS), Layer.ActivationFunctionType.SIGMOID)
                .fullyConnected(new Dimensions(1), Layer.ActivationFunctionType.SIGMOID)
                .build(Layer.OutputFunctionType.MEAN_SQUARES);
        // Test it with a different layer configuration and/or other activation functions

        final Trainer trainer = new AdadeltaTrainer(network, 0.95, 1E-6); // Try out with different trainers

        final BiFunction<Boolean, Boolean, Boolean> logicFunction = XOR; // Try out with other functions

        final Random random = new Random();

        // Training
        for(int i = 0; i < 10000; i++) // Increase the number of iterations for better results
        {
            final boolean a = random.nextBoolean(), b = random.nextBoolean();
            final boolean z = logicFunction.apply(a, b);

            final Volume input = new Volume(new Dimensions(2));
            input.set(0, doubleValue(a));
            input.set(1, doubleValue(b));

            final Volume expectedOutput = new Volume(new Dimensions(1));
            expectedOutput.set(0, doubleValue(z));

            trainer.train(input, expectedOutput);

            System.out.println("Iteration: " + trainer.getSeen() + "\tLoss: " + trainer.getLoss());
        }

        // Validation
        for(int i = 0; i < 1 << 2; i++)
        {
            final double a = doubleValue(booleanValue(i >> 1)), b = doubleValue(booleanValue(i & 1));

            final Volume input = new Volume(new Dimensions(2));
            input.set(0, a);
            input.set(1, b);

            network.forwardPropagation(input);

            System.out.println("a: " + a + "\tb: " + b + "\t=> z: " + network.getOutput().get(0));
        }
    }

    private static double doubleValue(boolean b)
    {
        return b ? 1.0 : 0.0;
    }

    private static boolean booleanValue(int i)
    {
        return i != 0;
    }
}