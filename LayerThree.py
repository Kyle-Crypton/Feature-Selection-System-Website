import os
from numpy import *
from mysk import write_data
from FeatureSelection import Bunch
from FeatureSelection import LayerThreeSelection

def LayerThree(fileName):
	with open(os.getcwd() + '/static/results/result_layer2.txt', 'r') as f:
		twoOutputData = eval(f.read())
	twoOutputData = Bunch(**twoOutputData)
	twoOutputData.resultIndex = Bunch(**twoOutputData.resultIndex)
	with open(os.getcwd() + '/static/results/twoRetainIndex.txt', 'r') as f:
		twoRetainIndex = eval(f.read())
	resultWriterPath = os.getcwd() + '/static/results/result.txt'
	resultWriterPath3 = os.getcwd() + '/static/results/result3.txt'
	resultWriter = open(resultWriterPath, 'a')
	resultWriter3 = open(resultWriterPath3, 'w')
	print("------------------------------Layer Three-------------------------------")
	resultWriter.write("\n------------------------------Layer Three-------------------------------\n")
	# resultWriter3.write("\n------------------------------Layer Three-------------------------------\n")
	threeOutputData = LayerThreeSelection(twoOutputData,twoOutputData.resultIndex,resultWriter,resultWriter3)
	print("The layer three rmse: %f " %threeOutputData.resultIndex.rmse)
	threeRetainIndex = []*len(threeOutputData.optIndex)
	for i in range(0,len(threeOutputData.optIndex)):
		threeRetainIndex.append(twoRetainIndex[threeOutputData.optIndex[i]])
	print("The three retain features are:")
	print threeRetainIndex
	resultWriter.write("\n---The three retained features are" + str(sorted(threeRetainIndex)))
	resultWriter3.write("\n---The three retained features are" + str(sorted(threeRetainIndex)))
	with open(os.getcwd() + '/static/results/result_layer3.txt', 'w') as f:
		f.write(str(threeOutputData))
	resultWriter.close()
	resultWriter3.close()
	write_data(fileName, 3, sorted(threeRetainIndex))

if __name__ == '__main__':
	LayerThree('data1.csv')