# 문제2)
# 자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 대여횟수에 영향을 주는 변수들을 골라 다중선형회귀분석 모델을 작성하시오.
# https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv

# 모델 학습시에 발생하는 loss를 시각화하고 설명력을 출력하시오.
# 새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대여횟수 예측결과를 콘솔로 출력하시오.

import tensorflow as tf
from scipy._lib.pyprima.common import history
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', None)

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv', parse_dates = ['datetime'])
# ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
print(data.head(2))

# feature
x_data = data[['datetime', 'workingday', 'temp', 'humidity', 'windspeed']]
x_data['month'] = x_data['datetime'].dt.month
x_data['hour'] = x_data['datetime'].dt.hour
x_data['rush_hour'] = (
    ((x_data['hour'] >= 7) & (x_data['hour'] <= 10)) |
    ((x_data['hour'] >= 17) & (x_data['hour'] <= 19))
) & (x_data['workingday'] == 1)
x_data['rush_hour'] = x_data['rush_hour'].astype(int)
x_data.drop(['datetime', 'workingday'], axis=1, inplace=True)
print(x_data.head(2))
print(x_data.describe())

# label
y_data = np.log(data['count'])
print(y_data.head(2))

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 123, shuffle = False)

model = Sequential()
model.add(Input(shape=(x_data.shape[1],)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=50, verbose=0, validation_split=0.1)
print('evaluate result : ', model.evaluate(x_test, y_test, verbose=0))

pred = model.predict(x_test)
print('r2_score : ', r2_score(y_test, pred))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()


temp = float(input("temp: "))
humidity = float(input("humidity (0~100): "))
windspeed = float(input("windspeed: "))
month = int(input("month (1~12): "))
hour = int(input("hour (0~23): "))
workingday = input("workingday (y/n): ")

rush_hour = int(
    ((7 <= hour <= 10) or (17 <= hour <= 19))
    and (workingday == 'y')
)

input_data = pd.DataFrame([[
    temp, humidity, windspeed, month, hour, rush_hour
]], columns=x_data.columns)

pred_log = model.predict(input_data)[0][0]
pred_count = np.exp(pred_log)

print("\n🎯 예측 결과")
print("예측 대여횟수 : ", int(pred_count))