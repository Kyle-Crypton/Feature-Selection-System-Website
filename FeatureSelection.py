"""
===================================================================
Feature Selection with four layers
===================================================================

"""
print(__doc__)
import os
import time
import numpy as np
from numpy import *
from mysk import load_data
from mysk import write_data
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge


class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __getstate__(self):
        return self.__dict__

###############################################################################
# import data
def ImportData(fileName):
	sampleData = load_data(fileName)
	return sampleData
	
###############################################################################
# data preproprocessing
def Normalize(sampleData):
	minMaxScaler = preprocessing.MinMaxScaler()
	X = minMaxScaler.fit_transform(sampleData.data)
	Y = minMaxScaler.fit_transform(sampleData.target)
	print "X: ", X
	print "Y: ", Y 
	return Bunch(X=X,Y=Y)
	
###############################################################################
# data preproprocessing
def Normalize(sampleData):
	minMaxScaler = preprocessing.MinMaxScaler()
	X = minMaxScaler.fit_transform(sampleData.data)
	Y = minMaxScaler.fit_transform(sampleData.target)
	print "X: ", X
	print "Y: ", Y 
	return Bunch(X=X,Y=Y)
	
###############################################################################
# Sparsity Evaluate	
def SparsityEvaluate(inputX, threshold):
	vt =  VarianceThreshold()
	vt.fit_transform(inputX)
	importance = vt.variances_
	indices = np.argsort(importance)[::-1]
	filtered = filter(lambda x: importance[x] <= threshold, indices)
	indices = filter(lambda x: importance[x] > threshold, indices)
	print importance
	return Bunch(indices=indices,filtered=filtered)
	
###############################################################################
# Correlation Evaluate	
def CorrelationEvaluate(dataX,dataY):
	nFeatures = len(dataX[0])
	coorArray = []*nFeatures
	for i in range(0, nFeatures):
		l = [x[i] for x in dataX ]
		coor = pearsonr(l,dataY)
		coorArray.append(abs(coor[0]))
	coorIndex = np.argsort(coorArray)
	return coorIndex

###############################################################################
# Random Forest	
def RandomForest(dataX,dataY):
	rf = RandomForestRegressor()
	rf.fit(dataX,dataY)
	importance = rf.feature_importances_
	indices = np.argsort(importance)
	return indices
	
###############################################################################
# Layer One
def LayerOneSelection(oneInputData,resultIndex,resultWriter, resultWriter1):
	threshold = 0.01
	running = True
	seRmse = resultIndex.rmse
	optResultIndex = resultIndex
	outputX = oneInputData.dataX
	trainX, trainY = oneInputData.dataX[oneInputData.trainIndex], oneInputData.dataY[oneInputData.trainIndex]
	testX, testY = oneInputData.dataX[oneInputData.testIndex], oneInputData.dataY[oneInputData.testIndex]
	resultWriter.write("The layer one input data: \n")
	resultWriter.write("samples: " + str(len(oneInputData.dataX)) + "features: " + str(len(trainX[0])) + "\n")
	resultWriter.write("train data: " + str(len(trainX)) + "test data: " + str(len(testX))+ "\n")
	resultWriter1.write("The layer one input data: \n")
	resultWriter1.write("	-------------------------------------\n")
	resultWriter1.write("		samples: " + str(len(oneInputData.dataX)) + "	features: " + str(len(trainX[0])) + "\n")
	resultWriter1.write("	-------------------------------------\n")
	resultWriter1.write("		train data: " + str(len(trainX)) + "	test data: " + str(len(testX))+ "\n")
	resultWriter1.write("	-------------------------------------\n")
	indices = [i for i in range(0,len(oneInputData.dataX[0]))]
	while running:
		seIndex = SparsityEvaluate(oneInputData.dataX,threshold)
		seTrainX = trainX[:,seIndex.indices]
		seTestX = testX[:,seIndex.indices]
		seTrainY = trainY
		seTestY = testY
		dataLen = len(seIndex.indices)
		print("ResultIndex: %.4f " %resultIndex.rmse)
		if dataLen > 0:
			sePredictY = SvrPrediction(seTrainX,seTrainY,seTestX)
			seResultIndex = modelEvaluation(seTestY,sePredictY)
			print("seResultIndex: %.4f " %seResultIndex.rmse)
			if seResultIndex.rmse <= seRmse:
				seRmse = seResultIndex.rmse
				optResultIndex = seResultIndex
				threshold = threshold + 0.01
				outputX = oneInputData.dataX[:,seIndex.indices]
				indices = seIndex.indices
			else:
				running = False # this causes the while loop to stop
		else: 
			running = False # this causes the while loop to stop
	resultWriter.write("\nThe predition result in layer one is:\n")
	resultWriter.write("rmse: " + str(optResultIndex.rmse) + " mae: " + str(optResultIndex.mae) + " r2: " + str(optResultIndex.r2) + "\n")
	resultWriter1.write("\nThe predition result in layer one is:\n")
	resultWriter1.write("	-------------------------------------\n")
	resultWriter1.write("		rmse: " + str(optResultIndex.rmse) + "\n")
	resultWriter1.write("	-------------------------------------\n")
	resultWriter1.write("		mae:  " + str(optResultIndex.mae) + "\n")
	resultWriter1.write("	-------------------------------------\n")
	resultWriter1.write("		r2:     " + str(optResultIndex.r2) + "\n")
	resultWriter1.write("	-------------------------------------\n")
	return Bunch(oneDataX=outputX,oneDataY=oneInputData.dataY,trainIndex=oneInputData.trainIndex,testIndex=oneInputData.testIndex,resultIndex=optResultIndex,indices=indices)

###############################################################################
# Layer Two
def LayerTwoSelection(twoInputData,resultIndex,resultWriter, resultWriter2):
	ceRmse = resultIndex.rmse
	outputX = twoInputData.oneDataX
	optResultIndex = resultIndex
	dataX = twoInputData.oneDataX
	dataY = twoInputData.oneDataY
	trainX, trainY = twoInputData.oneDataX[twoInputData.trainIndex], twoInputData.oneDataY[twoInputData.trainIndex]
	testX, testY = twoInputData.oneDataX[twoInputData.testIndex], twoInputData.oneDataY[twoInputData.testIndex]
	coorIndex = CorrelationEvaluate(dataX,dataY)
	optIndex = coorIndex
	resultWriter.write("The layer two input data: \n")
	resultWriter.write("samples: " + str(len(twoInputData.oneDataX)) + "features: " + str(len(trainX[0])) + "\n")
	resultWriter.write("train data: " + str(len(trainX)) + "test data: " + str(len(testX))+ "\n")
	resultWriter2.write("The layer two input data: \n")
	resultWriter2.write("	-------------------------------------\n")
	resultWriter2.write("		samples: " + str(len(twoInputData.oneDataX)) + "	features: " + str(len(trainX[0])) + "\n")
	resultWriter2.write("	-------------------------------------\n")
	resultWriter2.write("		train data: " + str(len(trainX)) + "	test data: " + str(len(testX))+ "\n")
	resultWriter2.write("	-------------------------------------\n")
	print("ResultIndex: %.4f " %resultIndex.rmse)
	for i in range(1,len(coorIndex)):
		coorIn = []*(len(coorIndex)-i)
		for j in range(i,len(coorIndex)):
			coorIn.append(coorIndex[j])
		cePredictY = SvrPrediction(trainX[:,coorIn],trainY,testX[:,coorIn])
		ceResultIndex = modelEvaluation(testY,cePredictY)
		print("ceResultIndex: %.4f " %ceResultIndex.rmse)
		if ceResultIndex.rmse <= ceRmse:
			ceRmse = ceResultIndex.rmse
			optIndex = coorIn
			optResultIndex = ceResultIndex
		else:
			break
	outputX = dataX[:,optIndex]
	resultWriter.write("\nThe predition result in layer two is:\n")
	resultWriter.write("rmse: " + str(optResultIndex.rmse) + " mae: " + str(optResultIndex.mae) + " r2: " + str(optResultIndex.r2) + "\n")
	resultWriter2.write("\nThe predition result in layer two is:\n")
	resultWriter2.write("	-------------------------------------\n")
	resultWriter2.write("		rmse: " + str(optResultIndex.rmse) + "\n")
	resultWriter2.write("	-------------------------------------\n")
	resultWriter2.write("		mae:  " + str(optResultIndex.mae) + "\n")
	resultWriter2.write("	-------------------------------------\n")
	resultWriter2.write("		r2:     " + str(optResultIndex.r2) + "\n")
	resultWriter2.write("	-------------------------------------\n")
	return Bunch(twoDataX=outputX,twoDataY=dataY,trainIndex=twoInputData.trainIndex,testIndex=twoInputData.testIndex,resultIndex=optResultIndex,optIndex=optIndex)

###############################################################################
# Layer Three
def LayerThreeSelection(threeInputData,resultIndex,resultWriter,resultWriter3):
	reRmse = resultIndex.rmse
	optResultIndex = resultIndex
	dataX = threeInputData.twoDataX
	dataY = threeInputData.twoDataY
	outputX = threeInputData.twoDataX
	trainX, trainY = threeInputData.twoDataX[threeInputData.trainIndex], threeInputData.twoDataY[threeInputData.trainIndex]
	testX, testY = threeInputData.twoDataX[threeInputData.testIndex], threeInputData.twoDataY[threeInputData.testIndex]
	coorIndex = RandomForest(dataX,dataY)
	optIndex = coorIndex
	resultWriter.write("The layer three input data: \n")
	resultWriter.write("samples: " + str(len(threeInputData.twoDataX)) + "features: " + str(len(trainX[0])) + "\n")
	resultWriter.write("train data: " + str(len(trainX)) + "test data: " + str(len(testX))+ "\n")
	resultWriter3.write("The layer three input data: \n")
	resultWriter3.write("	-------------------------------------\n")
	resultWriter3.write("		samples: " + str(len(threeInputData.twoDataX)) + "	features: " + str(len(trainX[0])) + "\n")
	resultWriter3.write("	-------------------------------------\n")
	resultWriter3.write("		train data: " + str(len(trainX)) + "	test data: " + str(len(testX))+ "\n")
	resultWriter3.write("	-------------------------------------\n")
	print("ResultIndex: %.4f " %resultIndex.rmse)
	for i in range(1,len(coorIndex)):
		coorIn = []*(len(coorIndex)-i)
		for j in range(i,len(coorIndex)):
			coorIn.append(coorIndex[j])
		rePredictY = SvrPrediction(trainX[:,coorIn],trainY,testX[:,coorIn])
		reResultIndex = modelEvaluation(testY,rePredictY)
		print("reResultIndex: %.4f " %reResultIndex.rmse)
		if reResultIndex.rmse <= reRmse:
			reRmse = reResultIndex.rmse
			optIndex = coorIn
			optResultIndex = reResultIndex
		else:
			break
	outputX = dataX[:,optIndex]
	resultWriter.write("\nThe predition result in layer three is:\n")
	resultWriter.write("rmse: " + str(optResultIndex.rmse) + " mae: " + str(optResultIndex.mae) + " r2: " + str(optResultIndex.r2) + "\n")
	resultWriter3.write("\nThe predition result in layer three is:\n")
	resultWriter3.write("	-------------------------------------\n")
	resultWriter3.write("		rmse: " + str(optResultIndex.rmse) + "\n")
	resultWriter3.write("	-------------------------------------\n")
	resultWriter3.write("		mae:  " + str(optResultIndex.mae) + "\n")
	resultWriter3.write("	-------------------------------------\n")
	resultWriter3.write("		r2:     " + str(optResultIndex.r2) + "\n")
	resultWriter3.write("	-------------------------------------\n")
	return Bunch(threeDataX=outputX,threeDataY=dataY,trainIndex=threeInputData.trainIndex,testIndex=threeInputData.testIndex,resultIndex=optResultIndex,optIndex=optIndex)


###############################################################################
# Layer Four
def LayerFourSelection(fourInputData,resultIndex,resultWriter,resultWriter4):
	pcaRmse = resultIndex.rmse
	optResultIndex = resultIndex
	nFeatures = len(fourInputData.threeDataX[0])
	n_components = []*nFeatures
	resultWriter.write("The layer four input data: \n")
	resultWriter.write("samples: " + str(len(fourInputData.threeDataX)) + "features: " + str(len(fourInputData.threeDataX[0])) + "\n")
	resultWriter4.write("The layer four input data: \n")
	resultWriter4.write("	-------------------------------------\n")
	resultWriter4.write("		samples: " + str(len(fourInputData.threeDataX)) + "	features: " + str(len(fourInputData.threeDataX[0])) + "\n")
	resultWriter4.write("	-------------------------------------\n")
	print("ResultIndex: %.4f " %resultIndex.rmse)
	if nFeatures > 1:
		for i in range(1,nFeatures):
			n_components.append(i)
		print n_components
		Cs = np.logspace(-4, 4, 3)
		Cr = np.logspace(-3, 3, 7)
		g = np.logspace(-2, 2, 5)
		trainX, trainY = fourInputData.threeDataX[fourInputData.trainIndex], fourInputData.threeDataY[fourInputData.trainIndex]
		testX, testY = fourInputData.threeDataX[fourInputData.testIndex], fourInputData.threeDataY[fourInputData.testIndex]
		svr = SVR(kernel='rbf')
		pca = decomposition.PCA()
		pipe = Pipeline(steps=[('pca', pca), ('svr', svr)])
		estimator = GridSearchCV(pipe,
							 dict(pca__n_components=n_components,
								  svr__C=Cs,svr__gamma=g))
		estimator.fit(trainX,trainY)
		pcaPredictY = estimator.predict(testX)
		pcaResultIndex = modelEvaluation(testY,pcaPredictY)
		print("pcaResultIndex: %.4f " %pcaResultIndex.rmse)
		if pcaResultIndex.rmse <= pcaRmse:
			pcaRmse = pcaResultIndex.rmse
			optResultIndex = pcaResultIndex
	else:
		print "cant pca ...."
	resultWriter.write("\nThe predition result in layer three is:\n")
	resultWriter.write("rmse: " + str(optResultIndex.rmse) + " mae: " + str(optResultIndex.mae) + " r2: " + str(optResultIndex.r2) + "\n")
	resultWriter4.write("\nThe predition result in layer three is:\n")
	resultWriter4.write("	-------------------------------------\n")
	resultWriter4.write("		rmse: " + str(optResultIndex.rmse) + "\n")
	resultWriter4.write("	-------------------------------------\n")
	resultWriter4.write("		mae:  " + str(optResultIndex.mae) + "\n")
	resultWriter4.write("	-------------------------------------\n")
	resultWriter4.write("		r2:     " + str(optResultIndex.r2) + "\n")
	resultWriter4.write("	-------------------------------------\n")
	return optResultIndex
	
###############################################################################
# SVR	
def SvrPrediction(trainX,trainY,testX):
	rbfSVR = GridSearchCV(SVR(kernel='rbf'), cv=5,
			   param_grid={"C": np.logspace(-3, 3, 7),
						   "gamma": np.logspace(-2, 2, 5)})
	rbfSVR.fit(trainX,trainY)
	predictY = rbfSVR.predict(testX)
	return predictY

###############################################################################
# KNN
def KnnPrediction(trainX,trainY,testX):
	knnRegr = neighbors.KNeighborsRegressor(n_neighbors = '3', weights = 'distance');
	knnRegr.fit(trainX, trainY)
	predictY = knnRegr.predict(testX)
	return predictY

###############################################################################
# Decision Tree
def DTPrediction(trainX,trainY,testX):
	dtRegr = DecisionTreeRegressor(max_depth=5)
	dtRegr.fit(trainX, trainY)
	predictY = dtRegr.predict(testX)
	return predictY

###############################################################################
# Gradient Boosting Regressor	
def GTBPrediction(trainX,trainY,testX):
	gtbRegr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
	gtbRegr.fit(trainX, trainY)
	predictY = gtbRegr.predict(testX)
	return predictY
	
###############################################################################
# Kernel Ridge
def GTBPrediction(trainX,trainY,testX):
	krrRegr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
	kr_rbf.fit(trainX, trainY)
	predictY = kr_rbf.predict(testX)
	return predictY



###############################################################################
# model evaluation
def modelEvaluation(testY,predictY):
	rmse = np.sqrt(mean_squared_error(testY,predictY))
	mae = mean_absolute_error(testY,predictY)
	r2 = r2_score(testY,predictY)	
	return Bunch(rmse=rmse,mae=mae,r2=r2)
	

###############################################################################
# Main Procedure
def feature_selection(fileName):
	filePath = os.getcwd() + '/static/results/'
	# new_path = os.path.join(filePath, str(time.time())) +'/'
	# print new_path
	# os.makedirs(new_path)
	# with open('outputWriterPath.txt', 'w') as f:
	# 	f.write(new_path+'total_result.txt')
	# with open('outputWriterPath.txt', 'r') as f:
	# 	outputWriterPath = f.read()
	outputWriter = open(filePath + 'total_result.txt',"w+")
	t0 = time.time()
	print "Step 1: improt original data!"
	inputData = ImportData(fileName)  #Load original data
	if inputData.features == 0:
		print "Load data failed, please reload again:"
		ImportData()
	else:
		print "Load data successfully!"
		print "The sample number is : ", inputData.sample
		print "The feature is : ", inputData.features
	outputWriter.write("The sample number is : "+ str(inputData.sample))
	outputWriter.write("\nThe feature is : "+ str(inputData.features))

	print "Step 2: data preprocessing by normalize!"	
	normalizeData = Normalize(inputData)  #Data normalization
	outputWriter.write("\nThe original data has been normalized\n")
	print "Step 3: Cross validation with four layers feature selection!"     #param  n_folds
	kf = KFold(inputData.sample, n_folds=2)
	foldNum = 1
	totalRmse = 0
	outputWriter.write(str(len(kf))+ " folds cross validation")
	for train_index, test_index in kf:
		print("\nFeature selection in fold: %d" %foldNum)
		resultFilePath = os.getcwd()+'/static/results/'
		# with open('resultWriterPath.txt', 'w') as f:
		# 	f.write(resultFilePath + 'result.txt')
		# with open('resultWriterPath.txt', 'r') as f:
		# 	resultWriterPath = f.read()
		resultWriter = open(resultFilePath + 'result.txt',"w+")
		resultWriter1 = open(resultFilePath + 'result1.txt', "w+")
		resultWriter2 = open(resultFilePath + 'result1.txt', "w+")
		resultWriter3 = open(resultFilePath + 'result1.txt', "w+")
		resultWriter4 = open(resultFilePath + 'result1.txt', "w+")
		kRmse = 0
		orgTrainX, orgTestX = normalizeData.X[train_index], normalizeData.X[test_index]
		orgTrainY, orgTestY = normalizeData.Y[train_index], normalizeData.Y[test_index]
		orgPredictY = SvrPrediction(orgTrainX,orgTrainY,orgTestX)
		orgResultIndex = modelEvaluation(orgTestY,orgPredictY)
		kRmse = orgResultIndex.rmse
		print("The predition rmse in original data is: %f" % kRmse )
		resultWriter.write("The original input data: \n")
		resultWriter.write("samples: " + str(len(normalizeData.X)) + "features: " + str(len(orgTrainX[0])) + "\n")
		resultWriter.write("train data: " + str(len(orgTrainX)) + "test data: " + str(len(orgTestX))+ "\n")
		resultWriter.write("\nThe predition result is:\n")
		resultWriter.write("rmse: " + str(orgResultIndex.rmse) + " mae: " + str(orgResultIndex.mae) + " r2: " + str(orgResultIndex.r2) + "\n")
		oneInputData = Bunch(dataX=normalizeData.X,dataY=normalizeData.Y,trainIndex=train_index,testIndex=test_index)
	
		print("\n------------------------------Layer One-------------------------------")
		resultWriter.write("\n------------------------------Layer One-------------------------------\n")
		# resultWriter1.write("\n------------------------------Layer One-------------------------------\n")
		oneOutputData = LayerOneSelection(oneInputData,orgResultIndex,resultWriter, resultWriter1)
		print("The layer one rmse: %f " %oneOutputData.resultIndex.rmse)
		print("The retain features are:")
		print(oneOutputData.indices)
		resultWriter.write("\n---The one retained features are" + str(sorted(oneOutputData.indices)))
		resultWriter1.write("\n---The one retained features are" + str(sorted(oneOutputData.indices)))
		with open(os.getcwd() + '/static/results/result_layer1.txt', 'w') as f:
			f.write(str(oneOutputData))
		resultWriter.close()
		resultWriter1.close()
		write_data(fileName.split('/')[-1], 1, sorted(oneOutputData.indices))
		break
	
		print("------------------------------Layer Two-------------------------------")
		resultWriter.write("\n------------------------------Layer Two-------------------------------\n")
		resultWriter2.write("\n------------------------------Layer Two-------------------------------\n")
		twoOutputData = LayerTwoSelection(oneOutputData,oneOutputData.resultIndex,resultWriter,resultWriter2)
		print("The layer two rmse: %f " %twoOutputData.resultIndex.rmse)
		twoRetainIndex = []*len(twoOutputData.optIndex)
		for i in range(0,len(twoOutputData.optIndex)):
			twoRetainIndex.append(oneOutputData.indices[twoOutputData.optIndex[i]])
		print("The retained features are:")
		print twoRetainIndex
		resultWriter.write("\n---The two retained features are" + str(twoRetainIndex))
		resultWriter2.write("\n---The two retained features are" + str(twoRetainIndex))
	
		print("------------------------------Layer Three-------------------------------")
		resultWriter.write("\n------------------------------Layer Three-------------------------------\n")
		resultWriter3.write("\n------------------------------Layer Three-------------------------------\n")
		threeOutputData = LayerThreeSelection(twoOutputData,twoOutputData.resultIndex,resultWriter,resultWriter3)
		print("The layer three rmse: %f " %threeOutputData.resultIndex.rmse)
		# print("The retained three features")
		# print threeOutputData.optIndex
		threeRetainIndex = []*len(threeOutputData.optIndex)
		for i in range(0,len(threeOutputData.optIndex)):
			threeRetainIndex.append(twoRetainIndex[threeOutputData.optIndex[i]])
		print("The three retain features are:")
		print threeRetainIndex
		resultWriter.write("\n---The three retained features are" + str(threeRetainIndex))
		resultWriter3.write("\n---The three retained features are" + str(threeRetainIndex))
	
		print("------------------------------Layer Four-------------------------------")
		resultWriter.write("\n------------------------------Layer Four-------------------------------\n")
		resultWriter4.write("\n------------------------------Layer Four-------------------------------\n")
		fourResult = LayerFourSelection(threeOutputData,threeOutputData.resultIndex,resultWriter,resultWriter4)
		print("The layer four rmse: %f " %fourResult.rmse)
		resultWriter.write("the final rmse is : " + str(totalRmse))
		resultWriter4.write("the final rmse is : " + str(totalRmse))
		totalRmse=totalRmse+ fourResult.rmse
		foldNum+=1
		break
	totalRmse=totalRmse/len(kf)

	print("Feature selection finished!")
	print("the average rmse is : %f \n" %totalRmse)
	t1 = time.time()
	outputWriter.write("the average rmse is : " + str(totalRmse))
	outputWriter.write("the total cost time is : " + str(t1-t0))
	outputWriter.close()
	resultWriter.close()

if __name__ == '__main__':
	feature_selection(os.getcwd() + '/static/uploadfiles/data1.csv')
