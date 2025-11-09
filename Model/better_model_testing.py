import numpy as np 
import matplotlib.pyplot as plt

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

y = np.array([[0],[0],[0],[1]])

np.random.seed(42)
weights_input_hidden = np.random.randn(2, 2) 
bias_hidden = np.zeros((1,2))

weights_hidden_output = np.random.randn(2,1)
bias_output = np.zeros((1,1))

# Sigmoid Activation Function
# ----------------------------
# Formula: σ(x) = 1 / (1 + e^(-x))
# Range: (0, 1)
# Commonly used to map values into probabilities
#
# Derivative (used in backpropagation):
# σ'(x) = σ(x) * (1 - σ(x))
#
# Where:
#   e = Euler's number (~2.71828)
#   σ(x) = sigmoid(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

#Training part?

learning_rate = 0.5

for epoch in range(20000):
    #Forward pass?
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = sigmoid(final_input)

    #Calculate error? Maybe this is after the output and compare then after that when back and take error in hand and change the bias as stuffs like that 
    error = y - output
    
    #Backpropagation, searched google and still doesn't understand
    #After mulling over a bit, the theory side of calculating the thing will make it much clearer 
    #This part will need to see the math calculation for better visualization
    d_output = error * sigmoid_derivative(output) 
    d_hidden = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    #If my theory is right, after backpropagation, the updated version of biases and weights will be updated here
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis= 0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print('Predicted output : \n', output)



def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    neuron_radius = v_spacing/4

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), 
                                neuron_radius, color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Optional label
            if n == 0:
                ax.text(left - 0.1, layer_top - m*v_spacing, f"X{m+1}", ha='center', va='center')
            elif n == len(layer_sizes) - 1:
                ax.text(right + 0.1, layer_top - m*v_spacing, f"y{m+1}", ha='center', va='center')

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n+1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
ax.axis('off')

draw_neural_net(ax, .1, .9, .1, .9, layer_sizes=[2, 2, 1])
plt.title("Simple Neural Network (2-2-1)")
plt.show()
    

