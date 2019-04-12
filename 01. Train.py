import Data
import Model
import keras
from time import time



CONST_savePath = './Model_Trained/'
CONST_epochs = 20
CONST_stepsPerEpoch = 50
CONST_networkAmount = 3



data = Data.Data()
data.load(path='./Data/SNOMET_CT.csv', column='Value')
#data.load(path='./Data/va-deid-processed.csv', column='Value')
# data.load(path='./Data/BP_Train.csv', column='Value')


#data.constructTrainingValidationData(validationRatio=0.1)
data.constructTrainingValidationData(validationSize=1000)

print (data.paddingLength)


data.createTrainWorkspace(CONST_savePath)


for networkIndex in range(CONST_networkAmount):

	model = Model.Model1(input_shape=data.paddingLength)
	modelSavePoints = keras.callbacks.ModelCheckpoint(CONST_savePath+'model'+str(networkIndex)+'.h5', 
		verbose=1, monitor='loss', save_best_only=True, mode='auto')
	tensorboard = keras.callbacks.TensorBoard(log_dir="./logs/{}".format(time()))


	model.fit_generator(
		generator = data.trainDataGenerator(batchSize=100),
		steps_per_epoch = CONST_stepsPerEpoch,
		epochs=CONST_epochs, 
		verbose=1, 
		callbacks=[modelSavePoints, tensorboard], 
		#validation_data=(data.X_Validation, data.Y_Validation)
		)

