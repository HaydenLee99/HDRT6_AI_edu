# 앙상블 학습 : 여러 개의 분류기를 생성하고, 그 예측을 결합함으로써 보다 정확한 예측 결과를 얻음.
# 강력한 모델 하나 보다는 약한 모델 여러개를 조합하여 더 정확한 예측 결과를 도출.

import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np

cancer = load_breast_cancer()
print(cancer.keys())

x = cancer.data
y= cancer.target     # 1:암환자, 0:정상

# 0과 1의 비율
counter = Counter()
total = sum(counter.values())
for cls, cnt in counter.items():
    print(f'class: {cls}:{cnt}개 ({cnt/total:.2%})')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12, stratify=y)
# stratify train, test 비율 유지(불균형 데이터 모델 평가시 왜곡 방지)
y_li = y.tolist()
ytr_lin = y_train.tolist()
yte_lin = y_test.tolist()

# 개별 모델 생성
logi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs', random_state=12))
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
tree = DecisionTreeClassifier(max_depth=5, random_state=12)

# 앙상블 모델
voting = VotingClassifier(estimators=[('LP', logi),('KNN', knn),('DT',tree)], voting='soft')

# 개별 모델 성능 확인
named_models = [('LP', logi),('KNN', knn),('DT',tree)]
for name, clf in named_models:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(f'{name} 정확도 : {accuracy_score(y_test, pred):.4f}')
    # LP 정확도 : 0.9912
    # KNN 정확도 : 0.9737
    # DT 정확도 : 0.8772

# voting 성능
voting.fit(x_train, y_train)
vpred = voting.predict(x_test)
print(f'Voting 분류기 정확도 : {accuracy_score(y_test, vpred):.4f}')
# Voting 분류기 정확도 : 0.9649

# 선택: 교차검증으로 안전성 확인 필요
cvfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=12)
cv_score = cross_val_score(voting,x,y,cv=cvfold, scoring='accuracy')
print(f'보팅 5겹 cv 정확도 평균 : {cv_score.mean():.4f} (표준편차 : +-{cv_score.std():.4f})')
# 보팅 5겹 cv 정확도 평균 : 0.9701 (표준편차 : +-0.0181)

# 모델 성능 지표 이해
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print('보팅 모델 상세 평가')
print(classification_report(y_test, vpred))
print('confusion Matrix', confusion_matrix(y_test, vpred))
print('roc_auc_score : ', roc_auc_score(y_test,voting.predict_proba(x_test)[:,1]))

