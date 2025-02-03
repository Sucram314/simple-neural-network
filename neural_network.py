import numpy as np
import pickle as pk
from collections.abc import Callable
from mnist import MNIST

MNIST_DIRECTORY = r"C:\Users\marcu\OneDrive\Desktop\Python Scripts\Bigger Projects\AI\neural_network\MNIST"

def load_data() -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    data = MNIST(MNIST_DIRECTORY)

    testing_images, testing_labels = data.load_testing()
    training_images, training_labels = data.load_training()

    testing_images = np.array(testing_images).T / 255.0
    testing_labels = np.array(testing_labels)

    training_images = np.array(training_images).T / 255.0

    training_labels = np.array(training_labels)

    return testing_images, testing_labels, training_images, training_labels

def sigmoid(x : np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def dSigmoid(x : np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

def ReLU(x : np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def dReLU(x : np.ndarray) -> np.ndarray:
    return (x > 0)

LEAKINESS = 0.1 #[0,1)
def leakyReLU(x : np.ndarray) -> np.ndarray:
    return x * (LEAKINESS + (1 - LEAKINESS) * (x > 0))

def dLeakyReLU(x : np.ndarray) -> np.ndarray:
    return LEAKINESS + (1 - LEAKINESS) * (x > 0)

def tanh(x : np.ndarray) -> np.ndarray:
    return np.tanh(x)

def dTanh(x : np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x)**2

def softmax(x : np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp / np.sum(exp, 0)

def one_hot(x : np.ndarray) -> np.ndarray:
    res = np.zeros((x.size, 10))
    res[np.arange(x.size), x] = 1
    return res.T

class Neural_Network:
    def __init__(self, 
                 layers : np.ndarray, 
                 activation : Callable[[np.ndarray],np.ndarray] = ReLU, 
                 activation_derivative : Callable[[np.ndarray],np.ndarray] = dReLU, 
                 final_activation : Callable[[np.ndarray],np.ndarray] = softmax) -> None:
        
        self.nlayers = layers.size
        self.layers = layers

        self.activation = activation
        self.activation_derivative = activation_derivative
        self.final_activation = final_activation

        self.weights : list[np.ndarray] = []
        self.biases : list[np.ndarray]  = []

        self.iteration : int = 1
        self.accuracy : float = 0

        for i in range(1, self.nlayers):
            pre = self.layers[i-1]
            cur = self.layers[i]

            self.weights.append(np.random.rand(cur,pre) * 2 - 1)
            self.biases.append(np.random.rand(cur,1) * 2 - 1)            

    def forward_propagate(self, input_layer : np.ndarray) -> tuple[list[np.ndarray],list[np.ndarray]]:
        result : np.ndarray = input_layer

        layers = [input_layer]
        activated = [input_layer]

        for i in range(self.nlayers - 1):
            result = self.weights[i].dot(result) + self.biases[i]
            layers.append(result)

            if i == self.nlayers - 2:
                result = self.final_activation(result)
            else:
                result = self.activation(result)

            activated.append(result)

        return layers, activated
    
    def backward_propagate(self, layers : list[np.ndarray], activated : list[np.ndarray], target : np.ndarray) -> tuple[list[np.ndarray],list[np.ndarray]]:
        num_examples = target.size

        dweights = []
        dbiases = []

        dcurrent = activated[-1] - one_hot(target)
        dweights.append(1 / num_examples * dcurrent.dot(activated[-2].T))
        dbiases.append(1 / num_examples * np.sum(dcurrent,1,keepdims=1))

        for i in range(self.nlayers - 2, 0, -1):
            dcurrent = self.weights[i].T.dot(dcurrent) * self.activation_derivative(layers[i])
            dweights.append(1 / num_examples * dcurrent.dot(activated[i-1].T))
            dbiases.append(1 / num_examples * np.sum(dcurrent,1,keepdims=1))

        return dweights, dbiases

    def update_parameters(self, dweights, dbiases, alpha=0.1) -> None:
        for i in range(self.nlayers - 1):
            self.weights[i] -= dweights[-i-1] * alpha
            self.biases[i] -= dbiases[-i-1] * alpha

        self.iteration += 1

    def predict(self, input_layer : np.ndarray) -> np.ndarray:
        current : np.ndarray = input_layer

        for i in range(self.nlayers - 1):
            current =  self.weights[i].dot(current) + self.biases[i]

            if i == self.nlayers - 2:
                current = self.final_activation(current)
            else:
                current = self.activation(current)

        return np.argmax(current,0)
    
    def evaluate(self, testing_images, testing_labels) -> float:
        print(f"Iteration {neural_network.iteration}")

        predictions = neural_network.predict(testing_images)
        self.accuracy = np.sum(predictions == testing_labels) / testing_labels.size
        print(f"Accuracy: {self.accuracy:.2%}")

        print()

        return self.accuracy

if __name__ == "__main__":
    CACHE = r"C:\Users\marcu\OneDrive\Desktop\Python Scripts\Bigger Projects\AI\neural_network" + "\\" + input("What model would you like to train?: ")

    print("Loading Data...")
    testing_images, testing_labels, training_images, training_labels = load_data()
    print("Loaded Data Successfully!\n")

    try:
        with open(CACHE,"rb") as f:
            neural_network : Neural_Network = pk.load(f)

        print("Resuming Training...\n")

    except:
        neural_network : Neural_Network = Neural_Network(np.array([784,10,10,10]))

        print("Beginning Training...\n")

    best = neural_network.accuracy

    while True:
        layers, activated = neural_network.forward_propagate(training_images)
        dweights, dbiases = neural_network.backward_propagate(layers, activated, training_labels)
        neural_network.update_parameters(dweights, dbiases, 0.05)

        if neural_network.iteration % 10 == 0:
            neural_network.evaluate(testing_images,testing_labels)

            if neural_network.accuracy >= best:
                best = neural_network.accuracy

                with open(CACHE,"wb") as f:
                    pk.dump(neural_network,f)