from NeuralNetwork import NeuralNetwork
import numpy as np

nn = NeuralNetwork()
input_arr = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
output_arr = np.array([[0, 1, 1, 0]]).T
nn.train(input_arr, output_arr,20)
print(nn.predict(np.array([0,0,0])))
