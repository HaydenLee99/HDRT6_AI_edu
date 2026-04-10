# [XGBoost 문제] 
# kaggle.com이 제공하는 'glass datasets'          testdata 폴더 : glass.csv
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7 가지의 label(Type)로 분리된다.
# glass.csv 파일을 읽어 분류 작업을 수행하시오.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 데이터 준비 및 확인
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/glass.csv")
data.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 214 entries, 0 to 213
# Data columns (total 10 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   RI      214 non-null    float64
#  1   Na      214 non-null    float64
#  2   Mg      214 non-null    float64
#  3   Al      214 non-null    float64
#  4   Si      214 non-null    float64
#  5   K       214 non-null    float64
#  6   Ca      214 non-null    float64
#  7   Ba      214 non-null    float64
#  8   Fe      214 non-null    float64
#  9   Type    214 non-null    int64  
# dtypes: float64(9), int64(1)

print(data['Type'].unique())    # [1 2 3 5 6 7]     연속형이 아니다.
x = data.drop('Type', axis=1)
y = data['Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# y를 연속형으로 라벨 인코딩 - 0~5
le = LabelEncoder() 
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

xgb_clf = xgb.XGBClassifier(
    booster='gbtree',
    max_depth=6,
    n_estimators=200,
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(x_train, y_train)

pred_xgb = xgb_clf.predict(x_test)
print(f'XGBClassifier acc : {accuracy_score(y_test, pred_xgb):.5f}')    
# XGBClassifier acc : 0.81395

print("\nClassification Report:")
print(classification_report(y_test, pred_xgb))
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.85      0.79      0.81        14
#            1       0.85      0.73      0.79        15
#            2       0.67      0.67      0.67         3
#            3       0.75      1.00      0.86         3
#            4       0.67      1.00      0.80         2
#            5       0.86      1.00      0.92         6

#     accuracy                           0.81        43
#    macro avg       0.77      0.86      0.81        43
# weighted avg       0.82      0.81      0.81        43

new_data = np.array([
    [1.52, 13.2, 3.6, 1.1, 72.3, 0.0, 8.8, 0.0, 0.0],
    [1.51, 13.5, 3.0, 1.4, 72.0, 0.6, 8.2, 0.0, 0.1]
])
pred_enc = xgb_clf.predict(new_data)
pred_label = le.inverse_transform(pred_enc)

print("\nnew data result:", pred_label)

import matplotlib.pyplot as plt
xgb.plot_importance(xgb_clf)
plt.show()