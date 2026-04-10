# RandomForest 분류 알고리즘 - adult dataset(성인 소득 예측 자료)
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline           # 전처리 + 모델을 하나로 묶어서 실행
from sklearn.compose import ColumnTransformer   # 컬럼별 전처리를 다르게 적용할 때 사용함
from sklearn.impute import SimpleImputer        # 결측치 처리시 사용
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# adult dataset 가져오기
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.frame
print(df.head(3))
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 48842 entries, 0 to 48841
# Data columns (total 15 columns):
#  #   Column          Non-Null Count  Dtype        Column 의미
# ---  ------          --------------  -----        ----------
#  0   age             48842 non-null  int64        나이
#  1   workclass       46043 non-null  category     고용 형태
#  2   fnlwgt          48842 non-null  int64        가중치
#  3   education       48842 non-null  category     학력(문자)
#  4   education-num   48842 non-null  int64        학력(숫자)
#  5   marital-status  48842 non-null  category     결혼 상태
#  6   occupation      46033 non-null  category     직업
#  7   relationship    48842 non-null  category     가족 내 역할
#  8   race            48842 non-null  category     인종
#  9   sex             48842 non-null  category     성별
#  10  capital-gain    48842 non-null  int64        자본 이익
#  11  capital-loss    48842 non-null  int64        자본 손실
#  12  hours-per-week  48842 non-null  int64        주당 근무시간
#  13  native-country  47985 non-null  category     출신 국가
#  14  class           48842 non-null  category     소득_(target)
# dtypes: category(9), int64(6)

# data 전처리

# target인 'class'열 encoding : '>50K'는 1로, '<=50K'는 0으로
df['class'] = df['class'].apply(lambda x:1 if '>50K' in x else 0)
print(df['class'].unique())

x = df.drop('class', axis=1)        # feature 선언
y = df['class']                     # label 선언

# feature 타입별 컬럼 분리 : 숫자형과 범주형
num_cols = x.select_dtypes(include=['int64']).columns.tolist()       # 숫자형
cat_cols = x.select_dtypes(include=['category']).columns.tolist()    # 범주형

# 숫자형 컬럼 전처리 파이프라인 - 스케일링
num_pipeline = Pipeline([   # 처리 항목들을 연결해 연속적으로 실행
    ('imputer', SimpleImputer(strategy='median')),             # 결측치는 중앙값으로 처리
    ('scaler', StandardScaler())                               # 표준화(평균 0, 표준편차 1)
])

# 범주형 컬럼 전처리 파이프라인 - 인코딩
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),      # 결측치는 최빈값으로 처리
    ('onehot', OneHotEncoder(handle_unknown='ignore'))         # Onehot 처리. (불가능한 경우 무시)
])

# 컬럼별 전처리 결합
preprocess = ColumnTransformer([
    ('num', num_pipeline, num_cols),        # 숫자형 컴럼에 num_pipeline 적용
    ('cat', cat_pipeline, cat_cols)         # 범주형 컴럼에 cat_pipeline 적용
])

# 전체 파이프라인 (전처리 후 모델 생성)
pipeline = Pipeline([
    ('prep', preprocess),       # 전처리
    ('model', RandomForestClassifier(random_state=12))      # 모델 생성
])

# 학습, 검증 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=12, stratify=y)

# hyper parameter tuning 범위 설정
param_grid = {
    'model__n_estimators':[100,200],             # 트리 개수
    'model__max_depth':[5,10,None],              # 트리 깊이
    'model__class_weight':[None, 'balanced']     # 클래스 불균형 보정 유무
}

# GridsearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

grid = GridSearchCV(
    pipeline,       # 전체 pipeline 사용
    param_grid,     # 탐색할 파라미터
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1       # 모든 CPU 사용
)

grid.fit(train_x, train_y)
print('최적 파라미터 : ', grid.best_params_)
print('최적 모델 : ', grid.best_estimator_)

# 예측
pred = grid.predict(test_x)
proba = grid.predict_proba(test_x)[:, 1]    # class 1에 대한 확률값

# 평가
print(f'정확도 : {accuracy_score(test_y, pred):.6f}')
print(f'roc_auc : {roc_auc_score(test_y, proba):.6f}')
print(f'classification_report :\n{classification_report(test_y, pred)}')