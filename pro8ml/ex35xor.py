# svm으로 xor 연산하기
x_data=[
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

# feature, label 분리
# feature = []
# label = []
# for p,q,r in x_data:
#     feature.append([p, q])
#     label.append(r)
# print(feature)
# print(label)
x_df = pd.DataFrame(x_data)
feature = np.array(x_df.iloc[:,:2])
label = np.array(x_df.iloc[:,-1])
print(feature)
print(label)

lmodel = LogisticRegression()       # 선형 분류모델
smodel = svm.SVC()                  # 선형, 비선형 분류모델

lmodel.fit(feature, label)
smodel.fit(feature, label)

pred1 = lmodel.predict(feature)
print('lmodel 예측값 : ',pred1)
pred2 = smodel.predict(feature)
print('smodel 예측값 : ',pred2)

acc1 = metrics.accuracy_score(label, pred1)
print('lmodel 정확도 : ', acc1)
acc2 = metrics.accuracy_score(label, pred2)
print('smodel 정확도 : ', acc2)