import keras
import Data
import numpy
import pandas
import Model
import math

testFileName = 'BP_Test.csv'
# testFileName = 'BP_Test_BigErrorRate.csv'

# Step 1. Read in networks
CONST_savePath = './Model_Trained/Model Exp 1 and 3/'
models = []
for networkIndex in range(3):
	models.append(keras.models.load_model(CONST_savePath+'model'+str(networkIndex)+'.h5'))
print ('Network loaded')

# Step 2. Apply on Training Data To decide threshold
data = pandas.read_csv('./Data/BP_Train.csv')

testString = data['Value'].tolist()

trainVariances = []
for i in range(1000):
# for i in range(len(testString)):
	print (i, '/', len(testString))
	testData = Data.Data().getCustomizeTestData(testString[i])
	predictions = []
	for networkIndex in range(len(models)):
		predictions.append(models[networkIndex].predict(testData)[0][0])
	variance = numpy.var(numpy.array(predictions))
	trainVariances.append(variance)

maxTrainVariance = max(trainVariances)
stdErrorTrainVariance = math.sqrt(numpy.var(numpy.array(trainVariances)))
threshold = maxTrainVariance+3*stdErrorTrainVariance
print (maxTrainVariance, stdErrorTrainVariance, threshold)
print ('Done threshold calculation.')
trainVariances = pandas.DataFrame(trainVariances, columns=['Variance']).to_csv('./Results/BP_Train.csv')



# Step 3. Apply threshold on application data
data = pandas.read_csv('./Data/'+testFileName)

testString = data['Value'].tolist()
labels = data['Label'].tolist()


result = []
TP=0
TN=0
FP=0
FN=0
for i in range(len(testString)):
	print (i, '/', len(testString))
	testData = Data.Data().getCustomizeTestData(testString[i])
	predictions = []
	for networkIndex in range(len(models)):
		predictions.append(models[networkIndex].predict(testData)[0][0])
	variance = numpy.var(numpy.array(predictions))
	

	flag = 1
	if variance>threshold:
		flag = 0

	if flag==1 and labels[i]==1:
		TP+=1
	if flag==0 and labels[i]==0:
		TN+=1
	if flag==1 and labels[i]==0:
		FP+=1
	if flag==0 and labels[i]==1:
		FN+=1

	result.append([testString[i], variance, flag, labels[i], maxTrainVariance, stdErrorTrainVariance, threshold])
	# if i>1000:
	# 	break

result = pandas.DataFrame(result, columns=['String', 'Variance', 'Prediction', 'Truth', 'MaxTrainVariance', 'StdErrorTrainVariance', 'Threshold'])
result.to_csv('./Results/'+testFileName)
print (result)



print ('TP:', TP)
print ('TN:', TN)
print ('FP:', FP)
print ('FN:', FN)

