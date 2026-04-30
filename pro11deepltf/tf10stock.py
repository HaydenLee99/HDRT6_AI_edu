# 주식 데이터로 다중선형회귀모델 작성
# 전날 데이터로 다음날 종가 예측

import tensorflow as tf
from nltk.tbl import feature
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.preprocessing import MinMaxScaler

stock = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/stockdaily.csv'
# 배열 자료로 읽기
datas = np.loadtxt(stock, delimiter=',', skiprows=1)
print(datas[:2], len(datas))     # (732, 5)

# feature
x_data = datas[:,:-1]
print(x_data.shape)     # (732, 4)
scaler = MinMaxScaler(feature_range=(0,1))
x_data = scaler.fit_transform(x_data)
print(x_data[:2])
print(scaler.inverse_transform(x_data[:2]))

# label : 종가
y_data = datas[:,-1]
print(y_data[:2])

print(x_data[0], y_data[0])
print(x_data[1], y_data[1])
x_data = np.delete(x_data, -1, axis=0)
y_data = np.delete(y_data, 0)
print(x_data[0], y_data[0])
print(x_data[1], y_data[1])

# train / test split 없이 모델
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

model.fit(x_data, y_data, epochs=200, verbose=0)
print('evaluate result : ', model.evaluate(x_data, y_data, verbose=0))
# evaluate result :  [62.609519958496094, 62.609519958496094]

pred = model.predict(x_data)

# 설명력 확인
from sklearn.metrics import r2_score
print('train / test split 안한 모델 설명력 확인 : ', r2_score(y_data, pred))
# train / test split 없이 설명력 확인 :  0.9938490210450255 -> 과적합 상태

# 시각화
plt.plot(y_data, 'b', alpha=0.5, label='실제값')
plt.plot(pred, 'r--', label='예측값')
plt.title('train / test split 안 한 모델')
plt.legend()
plt.show()

# train / test split 한 모델
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 123, shuffle = False)
model2 = Sequential()
model2.add(Input(shape=(4,)))
model2.add(Dense(units=1, activation='linear'))

model2.compile(optimizer='sgd', loss='mse', metrics=['mse'])

model2.fit(x_train, y_train, epochs=200, verbose=0, validation_split=0.15)
print('evaluate result : ', model2.evaluate(x_test, y_test, verbose=0))

pred2 = model2.predict(x_test)

# 설명력 확인
print('train / test split 한 모델 설명력 확인 : ', r2_score(y_test, pred2))

# 시각화
plt.plot(y_test, 'b', alpha=0.5, label='실제값')
plt.plot(pred2, 'r--', label='예측값')
plt.legend()
plt.title('train / test split 하고 validation split도 한 모델')
plt.show()

