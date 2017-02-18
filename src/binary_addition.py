## HAVE TO UPDATE THIS ---- NOT WORKING CURRRENTLY

# Base Class of a Simple Neural Network
class Network(object):
    def __init__(self, sizes):
        # Assumes one input layer, n -2 hidden layers, one output layer where 'n' is the number of layers
        self.num_layers = len(sizes)
        self.sizes = sizes
        # np.radnom.rand is a gaussian with mean 0, variance 1
        # For now we don't assume any biases
        # self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights_normal = [np.random.randn(y, x)/np.sqrt(x)
                               for x,y in zip(sizes[:-1], sizes[1:])]
        self.weights_hidden_hidden = [np.random.randn(x, x)/np.sqrt(x) for x in sizes[1:-1]]
        self.hidden_layer_values = [np.zeros(size) for size in sizes[1:-1]]
        self.layer_deltas = [np.zeros(size) for size in sizes[1:]]
        
    # This is the feed forward operation of the net
    def feedforward(self, a):
        ''' Returns the output of the network. If 'a' is the input '''
        for index in range(self.num_layers-1):
            w = self.weights_normal[index]
            ''' a' = sigmoid(w.a + b) '''
            if index==self.num_layers-2:
                a = sigmoid(np.dot(w, a))
            else:
                a = sigmoid(np.dot(w, a)) + np.dot(self.hidden_layer_values[index][-1], self.weights_hidden_hidden[index])
                self.hidden_layer_values[index].append(a.copy())
        return a
        
    # Train the data for epochs
    def train(self, train_data, binary_dim=8, eta=1, epochs=100):
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
        estimate = np.zeros_like(z)

        # Current Error
        error = 0
        for pos in range(binary_dim):
            # Generate input and output
            x = np.array([[a[binary_dim -pos -1], b[binary_dim -pos -1]]]).T
            y = np.array([[c[binary_dim -pos -1 ]]]).T

            z = self.feedforward(x)
            error += self.error(z, y)
            
            # Decoding the esimate
            estimate[binary_dim -pos -1] = np.round(predict[0][0])
        
        future_hidden_layers_delta = [np.zeros(size) for size in self.sizes[1:-1]]
        del_w = [np.zeros(w.shape) for w in self.weights_normal]
        del_w_h = [np.zeros(w_h.shape) for w_h in self.weights_hidden_hidden]
        # Update values
        for pos in range(binary_dim):
            x = np.array([[a[pos], b[pos]]]).T
            delta_w, delta_w_h, future_hidden_layers_delta = self.backprop(x, pos, future_hidden_layers_delta)
            del_w = [ nw + dnw for nw, dnw in zip(del_w, delta_w)]
            del_w_h = [ nw_h + dnw_h for nw_h, dnw_h in zip(del_w_h, delta_w_h)]
            
        self.weights_normal = [ w + eta*nw for w, nw in zip(self.weights_normal, del_w)]
        self.weights_hidden_hidden = [w_h + eta*nw_h for w_h, nw_h in zip(self.weights_hidden_hidden, del_w_h)]
    
    def backprop(self, x, pos, future_hidden_layers):
        current_hidden_layers = [self.hidden_layer_values[index][-pos-1] for index in range(1,self.num_layers-1)]
        prev_hidden_layers = [self.hidden_layer_values[index][-pos-2] for index in range(1,self.num_layers-1)]
        
        nabla_w = [np.zeros(w.shape) for w in self.weights_normal]
        nabla_w_hh = [np.zeros(w_h.shape) for w_h in self.weights_hidden_hidden]
        
        output_error = self.layer_deltas[-1][-pos-1]
        for l in xrange(2, self.num_layers):
            nabla_w[-l] = future_hidden_layers_delta[-l].dot(self.weights_hidden_hidden[-l])
        
    # Calculate the error at final layer
    def error(self, predict, actual):
        error = actual - predict
        self.layer_deltas[-1].append(error*sigmoid_derivative(predict))
        return np.abs(error[0])
