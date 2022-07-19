from linearAlgLibrary import * 
import math
import pickle


DIGIT_IMG_SIZE = 784
NUM_IMAGES = 60000
imagesPath = 'MNISTData/train-images-idx3-ubyte'
labelsPath = 'MNISTData/train-labels-idx1-ubyte'

def getImages():
    f = open(imagesPath, "rb")
    f.read(16)
    images = []
    for i in range(NUM_IMAGES):
        image = []
        for j in range(DIGIT_IMG_SIZE):
			##NOTE: we divide each images brightness by 256 to get strictly numbers
			##between 0-1
            image.append(float(ord(f.read(1)))/256)
        images.append(image)
    return images

def getLabels():
    l = open(labelsPath, "rb")
    l.read(8)
    labels = []
    for i in range(NUM_IMAGES):
        labels.append(ord(l.read(1)))
    return labels

"""
The image data is given to us in the following form:
	-a list of the 60,000 images, each of which is a list
	of length 784, with each element encoding the pixel brightness

Here, we convert the given image into a 784 x 1 vector.

NOTE: each vector is an instance of the Matrix class (see LinearAlgLibrary)
"""
def matrifyImageVector(image):
	imageMatrix = []
	for i in range(DIGIT_IMG_SIZE):
		imageMatrix.append([image[i]])
	return Matrix(imageMatrix)


"""
Convert all 60,000 images to Matrix form 
"""
def matrifyAllImages(images):
	matrifiedImages = []
	for img in images:
		matrifiedImages.append(matrifyImageVector(img))
	return matrifiedImages

def printPretty(image):
	strs = []
	currentStr = ""
	for i in range(784):
		if (i % 28 == 0 and i != 0) or (i == 783):
			if (i == 783):
				currentStr += str(math.ceil(image[i]/255))
			strs.append(currentStr)
			currentStr = ""
		currentStr += str(math.ceil(image[i]/255)) + "   "

	for _ in strs:
		print(_)



####(DEPRECATED) OLD 28x28 input matrix#####
""" def matrifyImageVector(image):
	currentImg = []
	currentRow = []
	for i in range(DIGIT_IMG_SIZE):
		if (i % 28 == 0 and i != 0) or (i == DIGIT_IMG_SIZE - 1):
			if (i == DIGIT_IMG_SIZE - 1):
				currentRow.append(image[i])
			currentImg.append(currentRow)
			currentRow = []
		currentRow.append(image[i])
	return Matrix(currentImg) """