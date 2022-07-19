from network import * 

testNet1 = Network((784,30,10))
result = testNet1.test(testImages, testLabels)
print(result)
