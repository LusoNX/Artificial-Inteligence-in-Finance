from pandas_datareader import data
import cufflinks as cf
import yfinance as yf
import plotly
import plotly.offline as offline
import datetime as dt
import matplotlib.pyplot as plt 
import websocket 
from datetime import datetime
import time
from tensorflow.keras.layers import Input, Dense,LSTM, SimpleRNN, Flatten,GRU,GlobalMaxPool1D, Embedding, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
import pandas as pd 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
import glob
import numpy as np 

path = r"YOUR PATH"
all_files = glob.glob(path + "/*.csv")
np.random.seed(7)


tickers_2 = ["ADA","ATOM","BTC","DOGE","DOT","ETH","LTC","SHIB"]

tickers = ["ADA-USD","ATOM-USD","BTC-USD","DOGE-USD","DOT-USD","ETH-USD","LTC-USD","SHIB-USD"]
stocks = ["RRC","AR","FB","MSFT","ESTE","BTC-USD","TSLA","MCD"]
df = pd.DataFrame()
ohlcv_data = {}
return_df = pd.DataFrame()


def get_data(stocks):
	for i in stocks:
		temp = yf.download(i,start="2018-04-17", end="2022-02-27",interval = "1d") ## Esta tem que ser a data minima visto os 
		df = pd.DataFrame(temp)
		df["returns"] = df["Close"].pct_change()
		df.to_csv("price_data_{}.csv".format(i))
		

	return return_df,ohlcv_data



def clean_columns(DF):
	df = DF.copy()
	#df.set_index("Volume",inplace = True)
	df = df.astype(str)
	df = df.replace("inf","0")
	#df = df.replace("-1","0")
	df = df.astype(float)
	df = df.fillna(0)
	#df = df.iloc[1:len(df)]
	return df

def get_returns(symbol):
	returns_dict = {}
	price_dict = {}
	volume_dict = {}
	vol_dict = {}
	trades_dict = {}
	for i,x in zip(all_files,tickers_2):
		df_v = pd.read_csv(i,index_col = 0,header = 0)
		returns_dict[x] = df_v["returns"]
		price_dict[x] = df_v["Close"].iloc[0]
		volume_dict[x] = df_v["Volume"].pct_change()
		vol = (df_v["High"] - df_v["Low"])/df_v["Low"]
		vol_dict[x] = vol
	df = returns_dict[symbol]
	volume = volume_dict[symbol]
	vol = vol_dict[symbol]
	volume = clean_columns(volume)

	#Sentiment data (Crypto Greed and Fear)
	df_sentiment = pd.read_csv("df_sentiment.csv",index_col = 0)
	df_sentiment = df_sentiment["sentiment_score"].pct_change()

	df_sentiment = df_sentiment.iloc[0:len(df)]



	price = price_dict[x]
	return df,volume,vol , price,df_sentiment





def transform_data(df):
	# normal values
	dataset = df.values
	dataset = dataset.reshape(-1,1)
	# Normalized Values
	#scaler_n = MinMaxScaler(feature_range = (0,1))
	#dataset_n = scaler.fit_transform(dataset)

	# Standardized Values
	scaler_s = StandardScaler()
	dataset_s = scaler_s.fit_transform(dataset)

	return dataset


def split_train_test(_dataset,train_size):
	dataset = _dataset.copy()
	train_size = int(len(dataset) * train_size)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	return train,test


# convert an array of values into a dataset matrix
# Cria uma matriz onde X são os retornos em t e Y os retornos em t+1
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0:dataset.shape[1]]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
		
	return np.array(dataX), np.array(dataY)



def lstm_model(x_train,y_train,x_test,y_test,lstm_units,look_back,layers,_epochs,_batch_size):
	model = Sequential()
	model.add(LSTM(lstm_units,input_shape = (4,look_back)))
	model.add(Dense(layers))
	model.compile(loss = "mean_squared_error", optimizer = "adam")
	r = model.fit(x_train,y_train,validation_data = (x_test,y_test),epochs = _epochs,batch_size=_batch_size, verbose = 0)
	print(model.summary())
	scores = model.evaluate(x_test, y_test, verbose=0)


	train_prediction = model.predict(x_train)
	test_prediction = model.predict(x_test)	

	plt.plot(r.history["loss"], label = "Loss")
	plt.plot(r.history["val_loss"], label = "Validation Loss")
	plt.plot(figsize=(10,6))
	plt.title("Y = Price Returns")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.xlim((0,_epochs))
	plt.legend()
	plt.show()

	return train_prediction,test_prediction

def price_pred(price,values,train_prediction,test_prediction,look_back):
	trainPredictPlot = np.empty_like(values)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(train_prediction)+look_back, :] = train_prediction	

	# shift test predictions for plotting
	testPredictPlot = np.empty_like(values)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(train_prediction)+(look_back*2)+1:len(values)-1, :] = test_prediction

	plt.plot(values)
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()
	return trainPredictPlot,testPredictPlot



def main():
	look_back = 3
	df,volume,vol,price,df_sentiment = get_returns("ATOM")
	price_data = transform_data(df)
	volume_data = transform_data(volume)
	vol_data = transform_data(vol)
	sentiment_score = transform_data(df_sentiment)
	data_app = np.append(price_data,volume_data,axis =1)
	data_app = np.append(data_app,vol_data,axis = 1)
	data_app = np.append(data_app,sentiment_score,axis = 1)
	data_app = data_app[1:,:]

	
	train,test = split_train_test(data_app,0.67)
	x_train,y_train = create_dataset(train,look_back)
	x_test,y_test = create_dataset(test,look_back)

	## No estado corrente os dados estão numa estrutura [sample,features]. sklearn input para a LSTM exige [sample,timestep,features]
	x_train = np.reshape(x_train,(x_train.shape[0],data_app.shape[1],x_train.shape[1]))
	x_test = np.reshape(x_test,(x_test.shape[0],data_app.shape[1],x_test.shape[1]))


	## Deploy the model
	train_prediction,test_prediction = lstm_model(x_train,y_train,x_test,y_test,20,look_back,1,100,5) #x_train,y_train,lstm_units,look_back,layers,_epochs,_batch_size)
	# Estimate the predictions
	price_data_pred = price_data[1:,:]
	trainPredictPlot,testPredictPlot = price_pred(price,price_data_pred,train_prediction,test_prediction,look_back)
	


main()







