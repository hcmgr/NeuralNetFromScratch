from network import * 

"""
Test runner
NOTE: must have mnist files loaded into repo
"""

testNet1 = Network((784,30,10))
result = testNet1.test(testImages, testLabels)
print(result)
