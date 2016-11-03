import os
from numpy import *
from mysk import write_data
from FeatureSelection import Bunch
from FeatureSelection import LayerTwoSelection

def LayerTwo(fileName):
	with open(os.getcwd() + '/static/results/result_layer1.txt', 'r') as f:
		oneOutputData = eval(f.read())
	oneOutputData = Bunch(**oneOutputData)
	oneOutputData.resultIndex = Bunch(**oneOutputData.resultIndex)
	resultWriterPath = os.getcwd() + '/static/results/result.txt'
	resultWriterPath2 = os.getcwd() + '/static/results/result2.txt'
	resultWriter = open(resultWriterPath, 'a')
	resultWriter2 = open(resultWriterPath2, 'w')
	print("------------------------------Layer Two-------------------------------")
	resultWriter.write("\n------------------------------Layer Two-------------------------------\n")
	# resultWriter2.write("\n------------------------------Layer Two-------------------------------\n")
	twoOutputData = LayerTwoSelection(oneOutputData,oneOutputData.resultIndex,resultWriter,resultWriter2)
	print("The layer two rmse: %f " %twoOutputData.resultIndex.rmse)
	twoRetainIndex = []*len(twoOutputData.optIndex)
	for i in range(0,len(twoOutputData.optIndex)):
		twoRetainIndex.append(oneOutputData.indices[twoOutputData.optIndex[i]])
	print("The retained features are:")
	print twoRetainIndex
	resultWriter.write("\n---The two retained features are" + str(sorted(twoRetainIndex)))
	resultWriter2.write("\n---The two retained features are" + str(sorted(twoRetainIndex)))
	with open(os.getcwd() + '/static/results/result_layer2.txt', 'w') as f:
		f.write(str(twoOutputData))
	with open(os.getcwd() + '/static/results/twoRetainIndex.txt', 'w') as f:
		f.write(str(twoRetainIndex))
	resultWriter.close()
	resultWriter2.close()
	write_data(fileName, 2, sorted(twoRetainIndex))
	
if __name__ == '__main__':
	LayerTwo('data1.csv')