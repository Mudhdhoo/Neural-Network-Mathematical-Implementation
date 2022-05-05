import numpy as np
from data import y_test

class Dense_Layer:

    #Initialize random weights and biases, define activation function of the layer. Mean of randomly initialized weights should be 0.
    def __init__(self, neurons, input_size, activation, activation_deriv):
        self.weights = np.random.rand(neurons, input_size) - 0.5
        self.biases = np.random.rand(1, neurons) - 0.5
        self.activation = activation
        self.activation_deriv = activation_deriv

    #Propagate forward
    def forward_propagation(self, inputs):
        self.pre_activation = np.dot(inputs,self.weights.T) + self.biases
        self.outputs = self.activation(self.pre_activation)
        self.inputs = inputs
        return self.outputs

    #Propagate back, the chain rule
    def back_propagation(self, output_derivative, activation_derivative, learning_rate):
        weights_derivative = np.dot(self.inputs.T,output_derivative * activation_derivative)
        bias_derivative = output_derivative * activation_derivative
        input_derivative = np.dot(output_derivative * activation_derivative, self.weights)

        #Update weight and biases using gradient descent
        self.weights -= weights_derivative.T * learning_rate
        self.biases -= bias_derivative * learning_rate

        return input_derivative

def relu(input):
    return np.maximum(0,input)

def relu_prime(input):
    return input > 0

def softmax(input):
    pre_exp = input - np.amax(input, axis = 1, keepdims = True)
    exp = np.exp(pre_exp)
    activation = exp/np.sum(exp, axis = 1, keepdims= True)
    return activation

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def mean_squared_error(predtictions, labels):
    loss = np.mean(np.square(np.subtract(predtictions,labels)))
    return loss

def mean_squared_error_prime(predictions,labels):
    dE_dY = 2 * np.subtract(predictions,labels) / np.size(predictions)
    return dE_dY

#Layers excluding input layer. Input information added manually as input for first hidden layer.
layers = [Dense_Layer(20,784,tanh,tanh_prime),
        Dense_Layer(20,20,tanh,tanh_prime),
        Dense_Layer(10,20,tanh,tanh_prime)]

def train(epochs,x,y):
    samples = len(x)
    for i in range(epochs):
        loss_sum = 0
        for j in range(samples):
            input = x[j]
            for layer in layers:
                input = layer.forward_propagation(input) # forward prop

            #print(input)
            loss_sum += mean_squared_error(input,y[j])
            error = mean_squared_error_prime(input, y[j])

            for layer in reversed(layers):
                activation_derivative = layer.activation_deriv(layer.pre_activation)
                error = layer.back_propagation(error, activation_derivative, 0.1) # Backprop
                
        #Mean loss in each epoch
        print('Epoch:', i, 'Loss:', loss_sum / samples)

def predict(input_data, index):
    for i in range(len(input_data)):
        input = input_data[i]
        for layer in layers:
            input = layer.forward_propagation(input)
    prediction = np.argmax(input)
   # print('Prediction:', prediction, 'Real Value:', y_test[index].tolist().index(1))
    return prediction, y_test[index].tolist().index(1)

# Function for testing accuracy
def accuracy(test):
    h = 0
    for i in range(len(test)):
        pred, real = predict(test[i], i)
        if pred == real:
            h += 1
    return h/len(test)
