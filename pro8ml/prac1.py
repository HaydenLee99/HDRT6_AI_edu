# [로지스틱 분류분석 문제1]
# 소득 수준에 따른 외식 성향을 나타내고 있다. 
# 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 

# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

import pandas as pd

data = pd.DataFrame()
data["요일"] = ["토","토","토","화","토","월","토","토","토","토","토","토","토","토","일","월","화","수","목","금","토","토","토","토","일","토","일","토"]
data["외식유무"] = [0,0,0,1,0,1,1,1,0,0,1,1,0,1,1,1,0,0,1,1,1,1,0,0,1,1,0,0]
data["소득수준"] = [57,39,28,60,31,42,54,65,45,37,98,60,41,52,75,45,46,39,70,44,74,65,46,39,60,44,30,34]

# print(data)
data = data[(data["요일"]=="토") | (data["요일"]=="일")]
print(data["요일"].unique())    # ['토' '일']
data.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 21 entries, 0 to 27
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype 
# ---  ------  --------------  ----- 
#  0   요일      21 non-null     object
#  1   외식유무    21 non-null     int64 
#  2   소득수준    21 non-null     int64 
# dtypes: int64(2), object(1)

import statsmodels.formula.api as smf
result = smf.logit(formula="외식유무 ~ 소득수준", data=data).fit()
# print(result.summary())

pred = result.predict(data["소득수준"])
print('예측값:', (pred >= 0.5).values.astype(int))
print('실제값:', data["외식유무"].values)

# confusion matrix
conf_table = result.pred_table()
print("confusion matrix\n", conf_table)
[tn, fp], [fn, tp] = conf_table

# Accuracy (정확도) : 전체 중에서 맞춘 비율
a = (tp + tn) / (tp + tn + fp + fn)
print("Accuracy (정확도) : ", round(a*100,2),"%")

# Precision (정밀도) : 맞다고 예측한 것 중 실제로 맞은 비율
p = (tp) / (tp + fp)
print("Precision (정밀도) : ", round(p*100,2),'%')

# Recall (재현율) : 실제로 맞는 것 중에서 맞춘 비율
r = (tp) / (tp + fn)
print("Recall (재현율) : ", round(r*100,2),'%')

# F1-score : Precision과 Recall의 조화평균
f1 = 2*p*r / (p+r)
print("F1-score : ", round(r,2))

mm = input('\n소득수준을 입력하세요.(1~100)\t')
input_pred = result.predict(pd.DataFrame({"소득수준":[int(mm)]}))
print(f"소득수준 <{mm}>에 따른 외식 여부 분류는 <{input_pred[0]:.0f}> 입니다.")