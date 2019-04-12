import keras
import Data
import numpy
import pandas
import Model


CONST_savePath = './Model_Trained/'


testString = ['20/20', '20/23', 'J1', 'CF', '-20', '20/@#', 'Done 4/27/17 LK', 'J1 with glasses', '6/6',  '#@$@!$%!@#$!@@$#', '34rweiflubernoislsjdnueksrkfsmdlkjwenlksadjn', '10/2016', 'doctor says he will be fine', 'E', '20/30 OU']
# testString = ['20/20', '80/120', 'cascja', 'doctor says ok']

 

models = []
for networkIndex in range(3):
	#model = Model.Model1(68)
	#model.load_weights(CONST_savePath+'model'+str(networkIndex)+'.h5')
	#models.append(model)
	
	models.append(keras.models.load_model(CONST_savePath+'model'+str(networkIndex)+'.h5'))
print ('Network loaded')


result = []
for i in range(len(testString)):
	testData = Data.Data().getCustomizeTestData(testString[i])
	predictions = []
	for networkIndex in range(len(models)):
		predictions.append(models[networkIndex].predict(testData)[0][0])
	variance = numpy.var(numpy.array(predictions))
	result.append([testString[i], variance])

result = pandas.DataFrame(result, columns=['String', 'Variance'])
result.sort_values(by=['Variance'], inplace=True, ascending=True)
print (result)






