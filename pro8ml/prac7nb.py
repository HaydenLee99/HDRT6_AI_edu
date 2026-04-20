# [GaussanNB 문제] 
# 독버섯(poisonous)인지 식용버섯(edible)인지 분류
# https://www.kaggle.com/datasets/uciml/mushroom-classification
# feature는 중요변수를 찾아 선택, label:class
# 참고 : from xgboost import plot_importance

import pandas as pd
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/mushrooms.csv")
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 8124 entries, 0 to 8123
# Data columns (total 23 columns):
#  #   Column                    Non-Null Count  Dtype 
# ---  ------                    --------------  ----- 
#  0   class                     8124 non-null   object
#  1   cap-shape                 8124 non-null   object
#  2   cap-surface               8124 non-null   object
#  3   cap-color                 8124 non-null   object
#  4   bruises                   8124 non-null   object
#  5   odor                      8124 non-null   object
#  6   gill-attachment           8124 non-null   object
#  7   gill-spacing              8124 non-null   object
#  8   gill-size                 8124 non-null   object
#  9   gill-color                8124 non-null   object
#  10  stalk-shape               8124 non-null   object
#  11  stalk-root                8124 non-null   object
#  12  stalk-surface-above-ring  8124 non-null   object
#  13  stalk-surface-below-ring  8124 non-null   object
#  14  stalk-color-above-ring    8124 non-null   object
#  15  stalk-color-below-ring    8124 non-null   object
#  16  veil-type                 8124 non-null   object
#  17  veil-color                8124 non-null   object
#  18  ring-number               8124 non-null   object
#  19  ring-type                 8124 non-null   object
#  20  spore-print-color         8124 non-null   object
#  21  population                8124 non-null   object
#  22  habitat                   8124 non-null   object
# dtypes: object(23)

# 종속변수(반응변수)는 class, 나머지 22개는 모두 입력변수(설명변수, 예측변수, 독립변수)

# 변수명 변수 설명
# class      edible = e, poisonous = p
# cap-shape    bell = b, conical = c, convex = x, flat = f, knobbed = k, sunken = s
# cap-surface  fibrous = f, grooves = g, scaly = y, smooth = s
# cap-color     brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y
# bruises        bruises = t, no = f
# odor            almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
# gill-attachment attached = a, descending = d, free = f, notched = n
# gill-spacing close = c, crowded = w, distant = d
# gill-size       broad = b, narrow = n
# gill-color      black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w, yellow = y
# stalk-shape  enlarging = e, tapering = t
# stalk-root    bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?
# stalk-surface-above-ring fibrous = f, scaly = y, silky = k, smooth = s
# stalk-surface-below-ring fibrous = f, scaly = y, silky = k, smooth = s
# stalk-color-above-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
# stalk-color-below-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o,pink = p, red = e, white = w, yellow = y
# veil-type      partial = p, universal = u
# veil-color     brown = n, orange = o, white = w, yellow = y
# ring-number none = n, one = o, two = t
# ring-type     cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
# spore-print-color black = k, brown = n, buff = b, chocolate = h, green = r, orange =o, purple = u, white = w, yellow = y
# population abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
# habitat       grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 범주형(문자) -> 수치 데이터
for col in df.columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

print(df.head(2))

# feature / label 분리
x = df.drop('class', axis=1)
y = df['class']
print(x.shape, y.shape) # (8124, 22) (8124,)

print('\n데이터 분리 : 학습용(train data), 검증용(test data) ----')
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, stratify=y, random_state=1
)
print(x_train.shape, x_test.shape)   # (5686, 22) (2438, 22)

# 중요 변수 찾기
print('\n중요 변수 찾기 ----')
xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    eval_metric='logloss',
    random_state=1
)
xgb_model.fit(x_train, y_train)

feat_impo = pd.DataFrame({
    'feature': x.columns,
    'importance': xgb_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feat_impo)

# 중요한 변수 선택
top_features = feat_impo['feature'].values[:5]
print('선택한 중요 변수 5개 :', top_features)

# 중요 변수만 사용
x_train_sel = x_train[top_features]
x_test_sel = x_test[top_features]

# GaussianNB 모델 학습
print('\n분류 모델 생성 ----')
model = GaussianNB()
model.fit(x_train_sel, y_train)

# 예측 및 평가
pred = model.predict(x_test_sel)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)
# 예측값 :  [0 1 0 1 1 0 0 0 0 0]
# 실제값 :  [0 1 0 1 1 0 0 0 0 0]

print(f"총 갯수: {len(y_test)}, 오류수: {(y_test != pred).sum()}")
# 총 갯수: 2438, 오류수: 356
print('accuracy score : ', accuracy_score(y_test, pred)) # 0.854
print('confusion matrix : \n', confusion_matrix(y_test, pred))
#  [[1149  114]
#  [ 242  933]]

# 교차 검증
cv_score = cross_val_score(model, x[top_features], y, cv=5)
print('교차 검증 점수 : ', cv_score)
# [0.70276923 0.70461538 0.71630769 0.86707692 0.91502463]
print('교차 검증 평균 정확도 : ', np.mean(cv_score))
# 교차 검증 평균 정확도 :  0.7811587722622205

# 중요변수 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plot_importance(xgb_model, ax=ax, max_num_features=10)
plt.title('버섯 데이터 중요 변수')
plt.show()

# bar그래프로 시각화
plt.figure(figsize=(10, 8))
plt.bar(feat_impo['feature'][:10], feat_impo['importance'][:10], color='b')
plt.xticks(rotation=45)
plt.xlabel('feature')
plt.ylabel('importance')
plt.title('중요 변수 상위 10개')
plt.tight_layout()
plt.show()
