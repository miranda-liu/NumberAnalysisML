import random
import numpy as np

class Network(object):
    def __init__(self, sizes): # defining a function that initializes a Network object
                               # sizes is a list that contains the number of neurons in each layer respectively
        self.num_layers = len(sizes) # setting num_layers to the length of sizes
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
                    # the biases are initialized randomly 
                    # the first layer is assumed to be the input layer, so we skip it hence 
                    #      looping through sizes starts at 1
                    # y is the length of the generated 2D array
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
                    # the weights are also initialized randomly
                    # the zip() pairs up the elements from sizes
                        # eg. first and second element are paired, second and third are paired
                    # the pairs from the zip() function are used for the randn(y, x) parameters
                    # y is the length of the generated 2D array; x is the number of floats generated for each 1D array inside
            # np.random.randn generates Gaussian distributions with mean 0 and standard deviation 1
                # Gaussian distribution: data is like a bell curve, symmetry about mean

    def feedforward(self, a): # returns the output of the network when a is the input
        for b, w in zip(self.biases, self.weights):
            # for each b in biases and w in weights
            # the zip(self.biases, self.weights) matches up each bias with a weight
            a = sigmoid(np.dot(w, a)+b)
                # using the sigmoid function
                # the np.dot returns the dot product of w and a, then adds b
        return a
    
    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # the stochastic gradient descent function
        # training_data is a list of tuples that represent the training input and the desired outputs
        # epochs --> each completion is an epoch
            # epochs is the number of epochs to train for
        # mini_batch_size is the size of the mini_batches used to sample
        # eta is the learning rate (n)
        # the result would be compared to test_data if provided
       
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test=len(test_data)

        for j in range(epochs):
            random.shuffle(training_data) # in each epoch, randomly shuffle the training data
            mini_batches = [
                training_data[k : k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
                # divides the training_data into mini-batches

            for mini_batch in mini_batches: # loop through mini_batches
                self.update_mini_batch(mini_batch, eta) # apply a single step of gradient descent
                                # updates the network weights and biases according to a single iteration of gradient descent
                                #   using only training data in mini_batch
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test));
            else:
                print("Epoch {} complete".format(j))
                # FILL/FIGURE OUT WHAT THIS DOES STILL
        
    def update_mini_batch(self, mini_batch, eta):
        # this method updates the network's weights and biases by applying gradient descent using backpropagation to a single mini batch
        # backpropagation is short for backward propagation of errors
        #      it is an algorithm for supervised learning of artificial neural networks with gradient descent
        #      given an artificial neural network and an error function, the method calculates the gradient of the error 
        #      function with respect to the neural network's weights 
        # the mini_batch is a list of tuples (x,y) and eta is the learning rate
        nabla_b = [np.zeros(b.shape) for b in self.biases]
            # nabla means vector??
            # np.zeroes() returns a new array of a given shape and type filled with zeros
            # in this case, an array of shape b is returned filled with zeros
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: # iterates through each input in mini_batch
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
                #backprop invokes the backpropagation algorithm which is a fast way of computing the gradient of the cost function
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
            for b, nb in zip(self.biases, nabla_b)]
    
    def evaluate(self, test_data): # returns the number of test outputs for which the neural network
                                        # outputs the correct result. The neural network's output is assumed
                                        # to be the index of whichever neuron in the final layer has the 
                                        # highest activation
        test_results = [(np.argmax(self.feedforward(x)),y)
            for (x,y) in test_data
            ]
        return sum(int(x==y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, y): # returns the vector of partial derivatives for the output activations
        return (output_activations-y)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
            # np.exp does e^(-z)
            # if z is a list, it'll do it for each z

def sigmoid_prime(z):# returns the derivative of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))


