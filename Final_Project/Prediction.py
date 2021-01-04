from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

import plotly.graph_objects as go

np.random.seed(1)
tf.random.set_seed(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

from compare import Compare

import datetime
def Timeset():
  y=int(input("\n \t year : "))
  m=int(input("\n \t month : "))
  d = int(input("\n \t day : "))
  start= datetime.datetime(y,m,d)
  end = datetime.datetime(y,m,d+1)
  return start, end

def Model_dev(X_train):
	model = Sequential()
	model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
	model.add(Dropout(rate=0.2))
	model.add(RepeatVector(X_train.shape[1]))
	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(rate=0.2))
	model.add(TimeDistributed(Dense(X_train.shape[2])))
	model.compile(optimizer='adam', loss='mae')
	return model

def Plot_graph(df, test_score_df, Anomaly):

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df['timestamp'], y=df['value'], name='Value'))
  fig.update_layout(showlegend=True, title='NAB/data/realTwitts/Twitter_volume_AAPL.csv')
  fig.show()

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=test_score_df['timestamp'], y=test_score_df['loss'], name='Test loss'))
  fig.add_trace(go.Scatter(x=test_score_df['timestamp'], y=test_score_df['threshold'], name='Threshold'))
  fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
  fig.show()

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=test_score_df['timestamp'], y=scaler.inverse_transform(test_score_df['value']), name='Value'))
  fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=scaler.inverse_transform(anomalies['value']), mode='markers', name='Anomaly'))
  fig.update_layout(showlegend=True, title='Detected anomalies')
  fig.show()

def Create_anomaly(anomalies):
  print(anomalies)
  Anomaly = pd.DataFrame(data=anomalies)
  print(Anomaly.head())


df = pd.read_csv('data/realTweets/Twitter_volume_AMZN.csv')
df = df[['timestamp', 'value']]
df['timestamp'] = pd.to_datetime(df['timestamp'])
#df['timestamp'].min(), df['timestamp'].max()

#

start, end = Timeset()
train, test = df.loc[df['timestamp'] <= start], df.loc[df['timestamp'] > end]
train.shape, test.shape

scaler = StandardScaler()
scaler = scaler.fit(train[['value']])

train['value'] = scaler.transform(train[['value']])
test['value'] = scaler.transform(test[['value']])

TIME_STEPS=288

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['value']], train['value'])
X_test, y_test = create_sequences(test[['value']], test['value'])

print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')

model = Model_dev(X_train)
model.summary() 

history = model.fit(X_train, y_train, epochs=4, batch_size=32, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend();

model.evaluate(X_test, y_test)

X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples');

threshold = np.max(train_mae_loss)
print(f'Reconstruction error threshold: {threshold}')

X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples');

test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['value'] = test[TIME_STEPS:]['value']



anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
#anomalies.shape
Anomaly = Create_anomaly(anomalies)
Plot_graph(df, test_score_df, Anomaly)
