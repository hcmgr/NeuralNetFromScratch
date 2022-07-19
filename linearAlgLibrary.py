"""
---------------

Since we're doing this thing from scratch, we may as well make our own linear
algebra library. Who needs numpy?

---------------
"""

import os
import random

"""
General matrix class capturing some transformations/functions that can be
performed on a matrix.
"""
class Matrix():
	"""
	matrix is a list of lists whereby:
		-each list is a row 
		-if the matrix is n x m, the list has n lists of length m 
	"""
	def __init__(self, matrix):

		if not self.validMatrixForm(matrix):
			print(f"{matrix} is an invalid matrix")
			print("Must be: list of n lists, each of length m (where matrix is n x m dimensions)")
			return 
		self.matrix = matrix

	""" 
	Returns True if matrix is in correct form, false otherwise:
	ie True iff:
		-matrix is a list of lists AND;
		-all rows are same length
	"""
	def validMatrixForm(self, matrix):
		if type(matrix) != list:
			return False

		rowLength = len(matrix[0])
		for row in matrix:
			if (type(row) != list or len(row) != rowLength):
				return False
		##valid form
		return True

	"""
	Returns the matrix in its original 'list of lists' form
	"""
	def getMatrix(self):
		return self.matrix
	
	"""
	Generates a matrix of the given dimensions, the values of which are randomly 
	generated from a gaussian distribution with the given parameters.

	params:
		n,m - dimensions of the matrix
		mu - mean of normal dist. (default = 0)
		sigma - standard deviation of normal dist. (default = 1)
	"""
	@staticmethod
	def generateMatrixRand(n, m, mu=0, sigma=1):
		if (n == 0 or m == 0):
			print("Zero-dimension matrix is invalid")
			return None
		return Matrix([[random.gauss(mu, sigma) for _ in range(m)] for _ in range(n)])

	"""
	Generates a matrix of the given dimensions, the values of which are all ones

	params:
		n,m - dimensions of the matrix
	"""
	@staticmethod
	def generateMatrixZeroes(n, m):
		if (n == 0 or m == 0):
			print("Zero-dimension matrix is invalid")
			return None
		return Matrix([[0 for _ in range(m)] for _ in range(n)])

	"""
	Returns the a n x m dimensions of the matrix in the form of a (n, m) tuple
	"""
	def getDims(self):
		##NOTE: we can be assured the matrix rows are all the same length (see constructor)
		return (self.getN(), self.getM())

	
	def getN(self):
		return len(self.matrix)


	def getM(self):
		return len(self.matrix[0])

	"""
	Converts a (row, col) matrix position to an index in a flattened
	1D matrix and returns the index

	eg: in a 3x3 matrix, (1,0) -> 3, (2,2) -> 5

	params: 
		row, col - (row, col) matrix position of element
		m - width of the matrix
	"""
	@staticmethod
	def getIndex(row, col, m):
		return col + row*m

	"""
	Returns the flattened index (see getIndex()) of the maximum element in the matrix
	"""
	@staticmethod
	def maxElIndex(matrix):
		maxIndex = 0
		maxValue = matrix.matrix[0][0]
		for row in range(matrix.getN()):
			for col in range(matrix.getM()):
				if matrix.matrix[row][col] > maxValue:
					maxIndex = Matrix.getIndex(row, col, matrix.getM())
					maxValue = matrix.matrix[row][col]
		return maxIndex

	"""
	Returns the transposed matrix 

	NOTE: the original matrix is unchanged and a new matrix is returned
	"""
	def transpose(self):
		newMatrix = []
		for col in range(self.getM()):
			newRow = []
			for row in range(self.getN()):
				newRow.append(self.matrix[row][col])
			newMatrix.append(newRow)
		return Matrix(newMatrix)

	"""
	Returns the matrix multiplied by the given scalar

	NOTE: the original matrix is unchanged and a new matrix is returned
	"""
	def scalarMult(self, scalar):
		newMatrix = []
		for row in range(self.getN()):
			newRow = []
			for col in range(self.getM()):
				newRow.append(self.matrix[row][col] * scalar)
			newMatrix.append(newRow)
		return Matrix(newMatrix)

	"""
	Performs matrix addition on this matrix and the other given matrix

	NOTE: the original matrices are unchanged and a new matrix is retured
	"""
	def add(self, other):
		##check dimensions are valid
		if self.getDims() != other.getDims():
			print("matrices must be exact same dimensions to perform matrix addition")
			return None
		
		##perform matrix addition
		newMatrix = []
		for row in range(self.getN()):
			newRow = []
			for col in range(self.getM()):
				newRow.append(self.matrix[row][col] + other.matrix[row][col])
			newMatrix.append(newRow)
		return Matrix(newMatrix)

	"""
	Performs matrix multiplication on this vector with the other given vector

	NOTE: both original matrices are unchanged and a new matrix is returned
	"""
	def mult(self, other):
		##check dimensions are valid
		if self.getDims()[1] != other.getDims()[0]:
			print("dimensions are invalid for the dot product to be peformed")
			return None

		##perform dot product calculation
		newMatrix = []
		for row in range(self.getN()):
			newRow = []
			for rightCol in range(other.getM()):
				dotProdSum = 0
				for leftCol in range(self.getM()):
					dotProdSum += (self.matrix[row][leftCol] * other.matrix[leftCol][rightCol])
				newRow.append(dotProdSum)
			newMatrix.append(newRow)
		return Matrix(newMatrix)

	"""
	Returns the hadamard product of this matrix with the other matrix

	NOTE: both original matrices are unchanged and a new matrix is returned
	"""
	def hadamardProd(self, other):
		##dimensions must be identical to compute the hadamard product
		if self.getDims() != other.getDims():
			print("dimensions must be identical to compute the hadamard product")
			return None
		
		newMatrix = []
		for n in range(self.getN()):
			newRow = []
			for m in range(self.getM()):
				newRow.append(self.matrix[n][m] * other.matrix[n][m])
			newMatrix.append(newRow)
		return Matrix(newMatrix)

	"""
	Applies the given function to every element of the matrix an returns
	the resulting matrix

	params:
		func : function to be applied

	NOTE: the original matrix is unchanged and a new matrix is returned
	"""	
	def apply(self, func):
		newMatrix = []
		for row in range(self.getN()):
			newRow = []
			for col in range(self.getM()):
				newRow.append(func(self.matrix[row][col]))
			newMatrix.append(newRow)
		return Matrix(newMatrix)

	"""
	Returns visual representation of the matrix
	"""
	def __repr__(self):
		matrixString = ""
		for row in self.matrix:
			matrixString += os.linesep
			rowString = ""
			for entry in row:
				rowString += str(entry) + " "
			matrixString += (rowString + os.linesep)
		return matrixString

def __main__():
	##testing
	randomMatrix = Matrix.generateMatrixRand(9, 1)
	print(randomMatrix)
	print(randomMatrix.maxElIndex())

if __name__ == '__main__':
	__main__()
