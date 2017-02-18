
# coding: utf-8

# In[600]:

# Import Libraries
import numpy as np
import copy


# In[601]:

# Set the random seed for reproducibility
# np.random.seed(2016)


# In[602]:

# Sigmoid Activation function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[603]:

# Sigmoid Derivative
def sigmoid_derivative(z):
    return z*(1-z)


# In[620]:

# Base Class of a Simple Neural Network
class Network(object):
    def __init__(self, sizes):
        # Assumes one input layer, 1 hidden layers, one output layer
        self.sizes = sizes
        self.num_layers = len(sizes)
        # np.radnom.rand is a gaussian with mean 0, variance 1
        # For now we don't assume any biases
        # self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights_0 = np.random.rand(sizes[0],sizes[1])/np.sqrt(sizes[1])
        self.weights_1 = np.random.rand(sizes[1],sizes[2])/np.sqrt(sizes[2])
        self.weights_h = np.random.rand(sizes[1],sizes[1])/np.sqrt(sizes[1])
        self.weights_0 = 2 * np.random.rand(sizes[0],sizes[1]) - 1
        self.weights_1 = 2 * np.random.rand(sizes[1],sizes[2]) - 1
        self.weights_h = 2 * np.random.rand(sizes[1],sizes[1]) - 1
        self.hidden_values = [np.zeros(sizes[1])]
        self.output_deltas = []
        
    # This is the feed forward operation of the net
    def feedforward(self, a):
        ''' Returns the output of the network. If 'a' is the input '''
        hidden_layer = sigmoid(np.dot(a, self.weights_0) + np.dot(self.hidden_values[-1], self.weights_h))
        self.hidden_values.append(copy.deepcopy(hidden_layer))
        return sigmoid(np.dot(hidden_layer, self.weights_1))
        
    # Train the data for epochs
    def train(self, train_data, binary_dim=8, eta=0.1, epochs=10000):
        for epoch in xrange(epochs):
            max_num=np.power(2,binary_dim)
            # Perform a simple addition (a = b + c)
            a_int = np.random.randint(max_num/2) # Integer version
            a = train_data[a_int] # Binary encode

            b_int = np.random.randint(max_num/2)
            b = train_data[b_int]

            # Actual Answer
            c_int = a_int + b_int
            c = train_data[c_int]

            # Let us store our best guess (binary encoding)
            estimate = np.zeros_like(c)
            
            # Current Error
            cerror = 0
            for pos in range(binary_dim):
                # Generate input and output
                x = np.array([ [a[binary_dim -pos -1], b[binary_dim -pos -1]]])
                y = np.array([[c[binary_dim -pos -1 ]]]).T

                z = self.feedforward(x)
                cerror += self.error(z, y)

                # Decoding the esimate
                estimate[binary_dim -pos -1] = np.round(z[0][0])

            future_hidden_layer_delta = np.zeros(self.sizes[1])
            
            del_w_0 = np.zeros_like(self.weights_0)
            del_w_1 = np.zeros_like(self.weights_1)
            del_w_h = np.zeros_like(self.weights_h)
            
            # Update values
            for pos in range(binary_dim):
                x = np.array([[a[pos], b[pos]]])
                
                hidden_layer = self.hidden_values[-pos -1]
                prev_hidden_layer = self.hidden_values[-pos -2]

                # Error at the output layer
                output_delta = self.output_deltas[-pos -1]
                
                # Error at hidden layer
                hidden_layer_delta = (np.dot(future_hidden_layer_delta, self.weights_h.T) + np.dot(output_delta ,self.weights_1.T)) * sigmoid_derivative(hidden_layer)
                
                del_w_1 += np.atleast_2d(hidden_layer).T.dot(output_delta)
                del_w_h += np.dot(np.atleast_2d(prev_hidden_layer).T, hidden_layer_delta)
                del_w_0 += np.dot(x.T, hidden_layer_delta)
                
                future_hidden_layer_delta = hidden_layer_delta
            
            self.weights_0 += (eta*del_w_0)
            self.weights_1 += (eta*del_w_1)
            self.weights_h += (eta*del_w_h)
            
            # print out progress
            if(epoch % int(epochs/10) == 0):
                print "Error:" + str(cerror)
                print "Pred:" + str(estimate)
                print "True:" + str(c)
                out = 0
                for index,x in enumerate(reversed(estimate)):
                    out += x*pow(2,index)
                print str(a_int) + " + " + str(b_int) + " = " + str(out)
                print "------------"
        
    # Calculate the error at final layer
    def error(self, predict, actual):
        output_error = actual - predict
        self.output_deltas.append((output_error)*sigmoid_derivative(predict))
        return np.abs(output_error[0])        


# In[621]:

# Generate some training data
def load_training_data(binary_dim=8):
    int2binary = {}
    max_num = np.power(2, binary_dim)
    binary = np.unpackbits(np.array([range(max_num)], dtype=np.uint8).T, axis=1)
    for i in xrange(max_num):
        int2binary[i] = binary[i]
    return int2binary


# In[622]:

binary_dim = 8
data = load_training_data(binary_dim=binary_dim)
net = Network([2, 16, 1])
net.train(data,binary_dim=binary_dim)

