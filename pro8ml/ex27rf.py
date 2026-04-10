# Random Forest, XG-Boost : 분류 알고리즘

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv")
df.info()
print(df.isnull().any())
print(df.shape)
df = df.dropna(subset=['Pclass','Age','Sex'])
print(df.shape)

df_x = df[['Pclass','Age','Sex']]
print(df_x.head(3))

# Sex열 label encoding(문자범주형 -> 정수형)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_x.loc[:,'Sex'] = encoder.fit_transform(df_x['Sex'])
print(df_x.head(3))

df_y = df['Survived']       # 1: 생존, 0: 사망
print(df_y.head(3))

# 학습, 검증 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(df_x,df_y,test_size=0.2, random_state=12)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

# 모델 생성
model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=12)
# n_estimators 결정트리수
model.fit(train_x,train_y)
pred = model.predict(test_x)
print('예측값 : ', pred[:5])
print('실제값 : ', np.array(test_y[:5]))
print('맞춘 개수 : ', sum(test_y == pred))
print('전체 대비 맞춘 비율: ', sum(test_y == pred)/len(test_y))
print('분류 정확도 : ', accuracy_score(test_y, pred))

# 교차검증(KFold)
cross_vali = cross_val_score(model, df_x, df_y, cv=5)
print(cross_vali)
print('교차 검증 평균 정확도 : ', np.round(np.mean(cross_vali), 5))

# 중요 변수 확인하기
print('특성(변수) 중요도 : ', model.feature_importances_)
# feature_importances_ : 각 특성이 예측에 기여한 정도(중요도)를 수치로 표현. 값의 총합은 1.0. 숫자가 클수록 중요도가 높음을 의미.

# 시각화
import matplotlib.pyplot as plt
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('Feature Importance score')
plt.ylabel('Features')
plt.yticks(np.arange(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()

# 전체 변수 대상으로 중요도 확인
# Survived와 무관한 정보인 PassengerId, Name인 제외
# 문자형 열인 Name, Ticket, Cabin Encoding 필요
df_imsi = df[['Pclass','Age','Sex','Fare','SibSp','Parch']]
df_imsi.loc[:,'Sex'] = encoder.fit_transform(df_imsi['Sex'])
train_x, test_x, train_y, test_y = train_test_split(
    df_imsi,df_y,test_size=0.2,random_state=12
)
model.fit(train_x, train_y)
importances = model.feature_importances_
# 컬럼명 + 중요도
feature_df = pd.DataFrame({
    'feature':df_imsi.columns,
    'importance':importances
}).sort_values(by='importance', ascending=False)
print(feature_df)

# 시각화
import seaborn as sns
plt.figure(figsize=(8,5))
sns.barplot(x='importance',y='feature', data=feature_df, orient='h')
plt.xlabel('Feature Importance score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
plt.close()