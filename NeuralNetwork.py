from numpy import dot, random, exp, floor 

class NeuralNetwork(): 
    synaptic_weights = None
    
    def __init__(self):
        print("Initialising Neural Network...")
        random.seed(7)
        self.synaptic_weights = random.random((3,1))
        print("Initialized Weights: \n", self.synaptic_weights) 

    def train(self, input_arr, output_arr, epochs=10):
        for epoch in range(epochs):
            output = self.calculate(input_arr)
            print("Epoch : ",epoch)
            error = output_arr - output 
            adjustment = dot(input_arr.T, error *
                             self.sigmoid_derivative(output))
            self.synaptic_weights += adjustment
        print("Training Completed...")

    def sigmoid(self,x):
        return 1/(1+exp(-x))

    def sigmoid_derivative(self, x):
        return x/(1-x)

    def calculate(self, input_arr):
        return self.sigmoid(dot(input_arr, self.synaptic_weights))

    def error_calculation(self, training_output, actual_output):
        # calculate average of accuracies
        pass
    
    def predict(self, input_arr):
        return self.calculate(input_arr)

