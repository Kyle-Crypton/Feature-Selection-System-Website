import os
from numpy import *
from FeatureSelection import Bunch
from FeatureSelection import LayerFourSelection

def LayerFour():
	with open(os.getcwd() + '/static/results/result_layer3.txt', 'r') as f:
		threeOutputData = eval(f.read())
	threeOutputData = Bunch(**threeOutputData)
	threeOutputData.resultIndex = Bunch(**threeOutputData.resultIndex)
	resultWriterPath = os.getcwd() + '/static/results/result.txt'
	resultWriterPath4 = os.getcwd() + '/static/results/result4.txt'
	resultWriter = open(resultWriterPath, 'a')
	resultWriter4 = open(resultWriterPath4, 'w')
	print("------------------------------Layer Four-------------------------------")
	resultWriter.write("\n------------------------------Layer Four-------------------------------\n")
	# resultWriter4.write("\n------------------------------Layer Four-------------------------------\n")
	fourResult = LayerFourSelection(threeOutputData,threeOutputData.resultIndex,resultWriter,resultWriter4)
	print("The layer four rmse: %f " %fourResult.rmse)
	resultWriter.close()
	resultWriter4.close()

if __name__ == '__main__':
	LayerFour()