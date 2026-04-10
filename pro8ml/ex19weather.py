# LogisticRegression - 이항 분류(sigmoid) - 날씨 예보
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
data.info()
# print(data.columns)
# ['Date', 'MinTemp', 'MaxTemp', 'Rainfall', 'Sunshine', 'WindSpeed', 'Humidity', 'Pressure', 'Cloud', 'Temp', 'RainToday', 'RainTomorrow']

data2 = pd.DataFrame()
data2 = data.drop(['Date', 'RainToday'], axis=1)

# RainTomorrow  Yes=1, No=0으로 하여 Binomial
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})
# print(data2['RainTomorrow'].unique())     # [1 0]

# RainTomorrow : 종속변수(범주형, label, class),    나머지열 : 독립변수(feature)

# 학습용(train), 검증용(test) 데이터 분리 - 과적합 방지
train_data, test_data = train_test_split(data2, test_size=0.3, random_state=42)
print(train_data.shape, test_data.shape)
print(train_data.head(3), '\n',test_data.head(3))

# 모델 생성
col_select = "+".join(train_data.columns.difference(['RainTomorrow']))      # 독립변수
# print(col_select) 
my_formula = 'RainTomorrow ~ ' + col_select
# model = smf.glm(formula=my_formula, data=train_data, family=sm.families.Binomial).fit()
model = smf.logit(formula=my_formula, data=train_data).fit()
print(model.summary())
#                  coef       P>|z|  
# -----------------------------------
# Intercept    219.3889       0.000  
# Cloud          0.0616       0.601  > 0.05
# Humidity       0.0554       0.049  
# MaxTemp        0.1746       0.516  > 0.05
# MinTemp       -0.1360       0.079  
# Pressure      -0.2216       0.000  
# Rainfall      -0.1362       0.082  
# Sunshine      -0.3197       0.006  
# Temp           0.0428       0.875  > 0.05
# WindSpeed      0.0038       0.906  > 0.05

print('예측값 : ', np.rint(model.predict(test_data).values[:5]))
print('실제값 : ', test_data['RainTomorrow'].values[:5])

# 분류 정확도
conf_mat = model.pred_table()
print('confusion matrix', conf_mat)

from sklearn.metrics import accuracy_score
print('분류 정확도 : ', (conf_mat[0][0] + conf_mat[1][1])/len(train_data))
print('분류 정확도 : ', accuracy_score(test_data['RainTomorrow'], np.rint(model.predict(test_data))))