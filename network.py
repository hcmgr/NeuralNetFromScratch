from linearAlgLibrary import *
from loadImages import *
import math

### RETRIEVE MNIST DATA ###
"""
-rawImages is a list of 60000 lists, each of which represents a 28x28 pixel image
-these are lists of 784 (28x28) pixel values between 0-255, 0 being black, 255 being white
-rawImages is converted into a list of Matrix instances (see LinearAlgLibrary)
-the network is trained on the first 50000 of these and tested on the last 10000
(trainingImages and testImages repesectively)
"""
rawImages = getImages()
trainingImages = matrifyAllImages(rawImages[:50000])
testImages = matrifyAllImages(rawImages[50000:])
trainingLabels = getLabels()[:50000]
testLabels = getLabels()[50000:]
print("loaded")

### NETWORK ###

## helper functions ##

"""The sigmoid activation function"""
def sigmoid(z):
	return 1.0/(1.0+math.exp(-z))

"""Derivative of the sigmoid function."""
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

class Network():

	def __init__(self, layerSizes):
		"""
		A tuple containing the sizes of each of the layers
		eg: a network with 10 inputs neurons, 30 hidden neurons
		and 5 output neurons, layerStrucutre = (10, 30, 5) 
		"""
		self.layerSizes = layerSizes

		"""
		The weights are stored as a list of matrices (one matrix per layer).
		This is an nxm matrix where n is the number of neurons in the lth 
		layer and m is the number of neurons in the l-1th layer.

		eg: in a (10, 30, 5) network, the hidden layers weights is
		represented by a 30 x 10 matrix

		NOTE: each matrix is an instance of the Matrix class in
		LinearAlgLibrary.py (the linear algebra library built for this project)
		
		NOTE: we can see that the jth row of the matrix gives us the 
		weights connected to the jth neuron in the lth layer
		"""
		self.weights = self.initWeights()
		#print("first set of weights: ", self.weights[0])

		"""
		The biases are stored as a list of vectors (one vector per layer).
		This is a nx1 vector where n is the number of neurons in the 
		lth layer. 
		"""
		self.biases = self.initBiases()
		#print("first set of biases: ", self.biases[0])

	def initWeights(self):
		weights = []
		for x, y in zip(self.layerSizes[:-1], self.layerSizes[1:]):
			weights.append(Matrix.generateMatrixRand(y, x))
		return weights

	def initBiases(self):
		biases = []
		for x in self.layerSizes[1:]:
			biases.append(Matrix.generateMatrixRand(x, 1))
		return biases


	def gradDescent(self, trainingData, numEpochs, miniSize, learnRate, testData=None):
		for i in range(numEpochs):
			miniBatches = self.createMiniBatches(trainingData, miniSize)
			for miniBatch in miniBatches:
				##calculate grad C
				##change weights and biases accordingly
				pass
			##output results of epoch
			result = self.test(testImages, testLabels)
			print(f"Epoch {i}: {result}/{len(testImages)}\n")

	"""
	Returns a list of lists, whereby each list is a mini-batch of image matrices
	"""
	def createMiniBatches(self, trainingData, size):
		return [[trainingData[j] for j in range(i, i + size)] for i in range(0, len(trainingData), size)]


	"""
	Given an input vector (x), feed forward through the network 
	to calculate the output vector (a)

	params:
		x (Matrix): n x 1 input vector, where n is number of input neurons
	"""
	def feedForward(self, x):
		aPrev = x
		for w, b in zip(self.weights, self.biases):
			##formula for the activation of a neuron: ((w * a) + b)##
			z = w.mult(aPrev).add(b)
			aPrev = z.apply(sigmoid)
		return aPrev

	def test(self, testImages, testLabels):
		results = [(Matrix.maxElIndex(self.feedForward(x)), y) for x, y in zip(testImages, testLabels)]
		return sum(int(x == y) for x, y in results)
