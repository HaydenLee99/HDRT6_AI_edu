# 다중 선형 회귀
# 자동차 연비 예측
# early stopping 사용

import tensorflow as tf
from nltk.tbl import feature
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/auto-mpg.csv', na_values='?')
del data['car name']
data = data.dropna()

data.drop(['cylinders', 'acceleration', 'model year', 'origin'], axis=1, inplace=True)
print(data.head())

# sns.pairplot(data[['mpg',  'displacement',  'horsepower',  'weight']], diag_kind='kde')
# plt.show()

# train test split
train_dataset = data.sample(frac=0.7, random_state=123)
test_dataset = data.drop(train_dataset.index)

# 표준화
# (요소 - 평균) / 표준편차
train_stat = train_dataset.describe()
train_stat.pop('mpg')
train_stat = train_stat.transpose()

def stdscal_func(x):
    return (x-train_stat['mean']) / train_stat['std']


st_train_data = stdscal_func(train_dataset)
st_train_data = st_train_data.drop('mpg', axis=1)

st_test_data = stdscal_func(test_dataset)
st_test_data = st_test_data.drop('mpg', axis=1)

train_label = train_dataset.pop('mpg')
print(train_label[:3])

test_label = test_dataset.pop('mpg')
print(test_label[:3])

# model
def build_model():
    network = Sequential([
        Input(shape=(3,)),
        Dense(units=32, activation='relu'),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='linear')
    ])
    optimizer = optimizers.Adam(learning_rate=0.001)
    network.compile(optimizer=optimizer, loss='mean_squared_error',
                    metrics=['mean_squared_error', 'mean_absolute_error'])
    return network

model = build_model()
model.summary()
# Model: "sequential"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense (Dense)                   │ (None, 32)             │           128 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_1 (Dense)                 │ (None, 16)             │           528 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_2 (Dense)                 │ (None, 1)              │            17 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 673 (2.63 KB)
#  Trainable params: 673 (2.63 KB)
#  Non-trainable params: 0 (0.00 B)

EPOCHS = 10000
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# baseline : 이 값에 도달하면 학습 종료
history = model.fit(x=st_train_data, y=train_label, epochs=EPOCHS, verbose=2, batch_size=32,
                    validation_split=0.2, callbacks=[early_stop])
# callbacks : early_stop point
df = pd.DataFrame(history.history)
print(df.head(3))
print(df.columns)

def plt_history(df):
    hist = df
    hist['epoch'] = history.epoch

    plt.figure(figsize = (8,14))

    plt.subplot(2,1,1)
    plt.xlabel('Epochs')
    plt.ylabel('MAE [mpg]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'train_MAE')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'val_MAE')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epochs')
    plt.ylabel('MSE [mpg]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label = 'train_MSE')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'val_MSE')
    plt.legend()
    plt.show()

plt_history(df)

# 모델 평가
from sklearn.metrics import r2_score
loss, mse, mae = model.evaluate(x=st_test_data, y=test_label, verbose=2)
print(f'loss={loss:.3f}, mse={mse:.3f}, mae={mae:.3f}')
print('결정 계수 : ', r2_score(test_label, model.predict(x=st_test_data)))

new_data = pd.DataFrame({'displacement':[300,400],  'horsepower':[120,150],  'weight':[2000,4000]})
new_data_st = stdscal_func(new_data)
new_data_pred = model.predict(x=new_data_st).ravel()
print('새 값 예측결과 : ', new_data_pred)












