import pandas as pd
import numpy as np
import sklearn.preprocessing as sp

df_csv = pd.read_csv("nikkei_averages.csv")

print("*** df_csv ***")
df_csv.info()
print(df_csv.head())
print(df_csv.tail())


# 読み込んだCSVを使いやすいように並び替え、整形する

import numpy as np

df_all = df_csv[["date","opening_price","high_price","low_price","close_price"]]
df_all = df_all.sort_values("date")
df_all = df_all.assign(id=np.arange(len(df_all)))
df_all = df_all.set_index("id")

print("*** df_all ***")
df_all.info()
print(df_all.head())
print(df_all.tail())



def build_train_data(df, learn_len, input_len):
  df_data_sets = []

  for i in range(0, learn_len):
    df_data = df[i:i+input_len+1]
    df_data_sets.append(df_data)

  x, y = [], []

  for df_data in df_data_sets:
    scaled_close_prices = np.array(df_data["scaled_close_price"])
    x.append(scaled_close_prices[:input_len])
    y.append(scaled_close_prices[input_len:])

  x = np.array(x).reshape(len(x), input_len, 1)
  y = np.array(y).reshape(len(y), 1)
  
  return x, y



from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def fit_model(x, y):
  length_of_sequence = len(x[0])
  in_out_neurons = 1
  n_hidden = 300

  batch_size = 256
  epoch = 1000

  model = Sequential()
  model.add(LSTM(n_hidden,
           batch_input_shape=(None, length_of_sequence, in_out_neurons),
           return_sequences=False))
  model.add(Dense(in_out_neurons))
  model.add(Activation("linear"))
  model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))

  history = model.fit(x, y,
                     batch_size=batch_size,
                     epochs=epoch,
                     validation_split=0.2,
                     callbacks=[EarlyStopping(patience=10, verbose=1)])
  
  return model, history



def predict(model, df, predict_len):
  results = []
  
  for i in range(0, predict_len):
    future = [[np.append(df_input[i:]["scaled_close_price"].as_matrix(), results)]]
    future = np.array(future).reshape(len(future), len(df), 1)
    
    result = model.predict(future)
    
    results.append(result[0][0])
  
  return results



# 学習データとして使用するデータを抽出する

from sklearn.preprocessing import MinMaxScaler

SIMULATION_LEN = 250
START_ID = -1
LEARN_LEN = 100
INPUT_LEN = 30
PREDICT_LEN = 10

df = df_all[START_ID-SIMULATION_LEN-LEARN_LEN-INPUT_LEN:START_ID]

scaler = MinMaxScaler()
close_price = np.array(df["close_price"]).reshape(len(df), 1)
scaled_close_price = scaler.fit_transform(close_price)
df = df.assign(scaled_close_price=scaled_close_price)

for i in range(0, PREDICT_LEN):
  df["predict_" + str(i)] = 0.0

print("*** df ***")
df.info()
print(df.head())
print(df.tail())



for i in range(0, SIMULATION_LEN):
  df_subset = df[i:i+LEARN_LEN+INPUT_LEN]
  print("*** df_subset ***")
  df_subset.info()
  print(df_subset[:1])
  print(df_subset[-1:])
  
  x, y = build_train_data(df_subset, LEARN_LEN, INPUT_LEN)
  model, history = fit_model(x, y)
  
  df_input = df_subset[-INPUT_LEN:]
  print("*** df_input ***")
  df_input.info()
  print(df_input[:1])
  print(df_input[-1:])
  
  results = predict(model, df_input, PREDICT_LEN)
  print("*** results ***")
  print(results)
  
  for j in range(0, PREDICT_LEN):
    row = i + LEARN_LEN + INPUT_LEN - 1
    column = df.columns.get_loc("predict_"+str(j))
    
    df.iat[row, column] = results[j]

  print("*** df ***")
  print(df.iloc[i+LEARN_LEN+INPUT_LEN-1])



df.to_csv("predict.csv")

