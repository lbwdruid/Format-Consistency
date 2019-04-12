import pandas
import numpy
import os
import sklearn


CONST_paddingLength=14
# CONST_paddingLength=68


class Data:
	
	def load(self, path, column):
		self.df = pandas.read_csv(path)
		self.df = self.df[column]
		self.df.dropna(inplace=True)

		self.paddingLength = self.df.map(len).max()*2


	def createTrainWorkspace(self, path):
		os.makedirs(path, exist_ok=True)


	def stringToListWithPadding(self, text, length):
		result = []
		try:
			for c in text:
				result.append(ord(c))
			while (len(result)<length):
				result.insert(0, 0)
			if len(result)>length:
				result = result[0:length]
			return result
		except:
			result = [0 for _ in range(length)]
			return result


	def reshapeInputForLSTM(self, data):
		data = numpy.array(data, dtype=numpy.float32)
		data = data.reshape(data.shape[0], data.shape[1], 1)
		return data

	def getTrainDataSampleDistribution(self, sampleSize):
		samples = self.df['VA'].sample(n=sampleSize, random_state=1)
		frequencyList = samples.value_counts()
		item = frequencyList.index.tolist()
		frequency = frequencyList.tolist()
	
		return item, frequency

	def trainDataGenerator(self, batchSize):
		while True:
			sampledX = self.X_Training.sample(n=batchSize).tolist()
			X_batch = [self.stringToListWithPadding(text=sample, length=self.paddingLength) for sample in sampledX]
			X_batch = self.reshapeInputForLSTM(X_batch)
			Y_batch = numpy.ones(X_batch.shape[0])

			yield X_batch, Y_batch

	
	def constructTrainingValidationData(self, validationSize=None, validationRatio=None):
		# Step 0. Calculate the validation size
		if validationRatio is not None:
			validationSize = round(self.df.size*validationRatio)
		
		# Step 1. Shuffle original data
		self.df = sklearn.utils.shuffle(self.df, random_state=0)
		
		if validationSize==0:
			self.X_Training = self.df
		# Step 2. Construct validation data
		if validationSize>0:
			self.X_Training = self.df.iloc[:-validationSize]
			self.X_Validation = self.df.iloc[-validationSize:]
			self.X_Validation = [self.stringToListWithPadding(text=sample, length=self.paddingLength) for sample in self.X_Validation.tolist()]
			self.X_Validation = self.reshapeInputForLSTM(self.X_Validation)
			self.Y_Validation = numpy.ones(self.X_Validation.shape[0])

	
	def getCustomizeTestData(self, inputString):
		result = [self.stringToListWithPadding(text=inputString, length=CONST_paddingLength)]
		result = self.reshapeInputForLSTM(result)

		return result





