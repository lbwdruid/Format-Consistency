import keras
import Data

def Model1(input_shape):
	
	model = keras.models.Sequential()
	model.add(keras.layers.LSTM(input_shape*5, return_sequences=True, input_shape=(None, 1)))
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.LSTM(input_shape*5, return_sequences=True))
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.LSTM(input_shape*5))
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model





# def Model1(input_shape):

# 	model = keras.models.Sequential()

# 	model.add(
# 		keras.layers.LSTM(
# 			100, 
# 			input_shape=(input_shape, 1),
# 			#recurrent_dropout=0.2,
# 			#dropout=0.2
# 			)
# 		)
# 	model.add(keras.layers.Dense(1))
# 	model.compile(loss='mean_squared_error', optimizer='adam')
	

# 	return model











def Model2():

	model = keras.models.Sequential()

	model.add(
		keras.layers.LSTM(
			1, 
			input_shape=(Data.CONST_paddingLength, 1),
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)
	model.add(keras.layers.Dropout(0.5))
	model.add(
		keras.layers.LSTM(
			1, 
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)



	model.add(
		keras.layers.LSTM(
			1, 
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)
	model.add(keras.layers.Dropout(0.5))
	model.add(
		keras.layers.LSTM(
			1, 
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)

	model.add(
		keras.layers.LSTM(
			1, 
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)
	model.add(keras.layers.Dropout(0.5))
	model.add(
		keras.layers.LSTM(
			1, 
			recurrent_dropout=0.2,
			dropout=0.2,
			)
		)

	
	model.add(keras.layers.Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	

	return model









def Model3():

	model = keras.models.Sequential()
	LSTM_num = 5

	model.add(
		keras.layers.LSTM(
			LSTM_num, 
			input_shape=(Data.CONST_paddingLength, 1),
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)
	model.add(keras.layers.Activation('relu'))
	model.add(
		keras.layers.LSTM(
			LSTM_num, 
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)



	model.add(
		keras.layers.LSTM(
			LSTM_num, 
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)
	model.add(keras.layers.Activation('relu'))
	model.add(
		keras.layers.LSTM(
			LSTM_num, 
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)

	model.add(
		keras.layers.LSTM(
			LSTM_num, 
			recurrent_dropout=0.2,
			dropout=0.2,
			return_sequences=True
			)
		)
	model.add(keras.layers.Activation('relu'))
	model.add(
		keras.layers.LSTM(
			LSTM_num, 
			recurrent_dropout=0.2,
			dropout=0.2,
			)
		)


	
	model.add(keras.layers.Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	

	return model









