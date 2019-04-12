import keras
import Data
import numpy
import pandas


CONST_savePath = './Model_Trained/'


#testString = ['20/30 OU', 'OD 20/30+2, OS 20/20-1', 'vision today doing well, 20/20, plan to follow up in 2 weeks']
testString = ['Able to CF and read 20/800 on near card but poor effort']














# Step 1. Load model
models = []
for networkIndex in range(10):
	models.append(keras.models.load_model(CONST_savePath+'model'+str(networkIndex)+'.h5'))
print ('Network loaded')

# Step 2. Calculate the distribution in the ground truth

data = Data.Data()
data.load('./Data/VA-extracted.csv')
item, frequency = data.getTrainDataSampleDistribution(sampleSize=10000)

itemVariance = []
varianceTraining = []
varianceTrainingFrequency = []

for dataIndex in range(len(item)):
	testData = Data.Data().getCustomizeTestData(item[dataIndex])

	predictions = []	
	for networkIndex in range(10):
		predictions.append(models[networkIndex].predict(testData)[0][0])

	variance = numpy.var(numpy.array(predictions))
	itemVariance.append(variance)
	
	if variance not in varianceTraining:
		varianceTraining.append(variance)
		varianceTrainingFrequency.append(0)
	varianceIndex = varianceTraining.index(variance)
	varianceTrainingFrequency[varianceIndex]+=1


'''
for i in range(len(varianceTraining)):
	print (varianceTraining[i], varianceTrainingFrequency[i])
'''

	





for dataIndex in range(len(testString)):
	result = []
	testStr = testString[dataIndex]
	print (testStr)
	
	minVariance = 100000
	minIndex = 0
	varianceList = []
	resultList = []
	result = []
	
	for i in range(len(testStr)):
		for j in range(i+1, len(testStr)+1):
			testData = testStr[i:j]
			testDataForNeuralNetwork = Data.Data().getCustomizeTestData(testData)

			predictions = []
			for networkIndex in range(10):
				predictions.append(models[networkIndex].predict(testDataForNeuralNetwork)[0][0])

			variance = numpy.var(numpy.array(predictions))
			'''
			if variance in varianceTraining:
				varianceIndex = varianceTraining.index(variance)
				variance/=varianceTrainingFrequency[varianceIndex]
			'''
			
			if testData in item:
				variance/=frequency[item.index(testData)]
			result.append([testData, variance])

	result = pandas.DataFrame(result, columns=['String', 'Variance'])
	result.sort_values(by=['Variance'], inplace=True, ascending=True)

			
	print(result)
	print()
