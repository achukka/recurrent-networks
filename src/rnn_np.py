import numpy as np

# Tanh activation function
def tanh(z):
    return np.tanh(z)

# Softmax function
def softmax(z):
    zt = np.exp(z - np.max(z))
    return zt/ np.sum(zt)

class RNNnp:
    # Initializes a Recurrent Neural Network with the provided parameters
    def __init__(self, word_dim=8000, hidden_dim=100, bptt_truncate=4):
        '''
            word_dim   - Vocabulary Size
            hidden_dim - Dimension (number of units) of the hidden layer
            
        '''
        # Helper Variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        # Randomly Initialize the parameters
        self.U = np.random.rand(hidden_dim, word_dim)/np.sqrt(word_dim)
        self.V = np.random.rand(word_dim, hidden_dim)/np.sqrt(hidden_dim)
        self.W = np.random.rand(hidden_dim, hidden_dim)/np.sqrt(hidden_dim)
    
    def forward_propagation(self, x):
        '''
            Defines the forward propagation of the net
            Assumes 'x' is a one-hot vector, where the index of word is 1, and rest all are zeros
            Example: if word 'network' occurs at index 10 and VOCABULARY SIZE is 8000, then
                     vector for 'network' would be of dimension 8000 and contains '1' at index 10, rest all are zeros
        '''
        # Number of timesteps
        timesteps = len(x)
        # Save all the hidden states, since we need them later
        # Add an additional element for the initial hidden unit and set it to '0'
        hs = np.zeros((timesteps +1 , self.hidden_dim))
        hs[-1] = np.zeros(self.hidden_dim)
        
        # Again save the outputs at each time step
        out = np.zeros((timesteps+1, self.word_dim))
        
        # Now propogate the net for each time step
        for timestep in  xrange(timesteps):
            # Indexing U by x[t]
            hs[timestep] = tanh(self.U[:, x[timestep]]+ self.W.dot(hs[timestep-1]))
            out[timestep] = softmax(self.V.dot(hs[timestep]))
        return [out, hs]
    
    def predict(self, x):
        '''
            Perform forward propagation and returns the index of highest score
        '''
        out, hs = self.forward_propagation(x)
        return np.argmax(out, axis=1)

    def calculate_total_loss(self, x, y):
        '''
            Calculates the total loss for the network predictions
        '''
        loss = 0
        # For each sentence
        for index in xrange(len(y)):
            out, hs = self.forward_propagation(x[i])
            # Check the 'correct' words
            error_words = out[xrange(len(y[i])), y[i]]
            # Add it to loss 'entropy loss'
            loss += -1 * np.sum(np.log(error_words))
        return loss
    
    def calculate_loss(self, x, y):
        # Divides the total loss by number of training examples
        loss = self.calculate_total_loss(x, y)
        N = np.sum((len(y_i)) for y_i in y)
        return loss/N