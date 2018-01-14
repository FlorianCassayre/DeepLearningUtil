package me.cassayre.florian.dpu;

import static org.junit.Assert.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.junit.Test;

import me.cassayre.florian.dpu.layer.Layer;
import me.cassayre.florian.dpu.layer.Layer.ActivationFunctionType;
import me.cassayre.florian.dpu.layer.Layer.OutputFunctionType;
import me.cassayre.florian.dpu.network.Network;
import me.cassayre.florian.dpu.network.architecture.FeedForwardNetwork;
import me.cassayre.florian.dpu.util.volume.Dimensions;

public class TestSerializable {

	@Test
	public void test() {
		
		//example network
		Network network = new FeedForwardNetwork.Builder(new Dimensions(2))
                .fullyConnected(new Dimensions(10), Layer.ActivationFunctionType.SIGMOID)
                .fullyConnected(new Dimensions(1), Layer.ActivationFunctionType.SIGMOID)
                .build(Layer.OutputFunctionType.MEAN_SQUARES);
		
		byte[] serialized = null;
		try (ByteArrayOutputStream baos = new ByteArrayOutputStream();ObjectOutputStream oos = new ObjectOutputStream(baos)){
			oos.writeObject(network);
			oos.flush();
			serialized = baos.toByteArray();
		} catch (Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		
		try(ByteArrayInputStream bais = new ByteArrayInputStream(serialized); ObjectInputStream ois = new ObjectInputStream(bais)){
			Network newNetwork = (Network) ois.readObject();
		}catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		
	}

}
