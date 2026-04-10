# 🔹 과적합(Overfitting) 문제
# - 결정 트리는 데이터를 계속 분할하면서 '규칙을 완벽히 맞추려 함'
# - 결과: 학습 데이터에는 정확하지만, 새로운 데이터에는 성능 저하
# - 즉, 모델이 '노이즈까지 학습' → 일반화 성능 감소

# 🔹 과적합 방지 방법
# 1) 트리 깊이 제한(max_depth)
#    - 트리가 너무 깊어지면 데이터에 과적합
# 2) 최소 샘플 수 제한(min_samples_split, min_samples_leaf)
#    - 너무 작은 그룹으로 분할되는 것을 방지
# 3) 가지치기(pruning)
#    - 불필요한 분기 제거 → 간단한 트리 유지

# 🔹 train_test_split : 일반화 성능 향상
# - 학습용(train)과 검증용(test) 데이터를 나누는 함수
# - 목적: 학습 데이터로 모델 학습 → 검증 데이터로 성능 평가
# - 일반적인 비율: train 70~80%, test 20~30%
# - sklearn.model_selection.train_test_split 사용
# 🔹 예시
# X, y = 데이터와 레이블
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔹 역할
# - 모델이 '본 적 없는 데이터'에서 어떻게 성능을 내는지 평가 가능
# - 과적합 방지와 성능 검증의 첫 단계

# 🔹 K-fold 교차검증(K-fold Cross Validation) : 안정적 평가
# - 데이터셋을 K개의 폴드(fold)로 나눔
# - K번 반복:
#    - 한 폴드를 검증용으로, 나머지 K-1개 폴드를 학습용으로 사용
# - 결과: 데이터셋 전체에 대한 평균 성능 평가 가능
# - 장점: 특정 train/test 분할에 의한 편향 감소, 안정적인 평가

# 🔹 GridSearchCV : 최적의 하이퍼 파라미터 검색
# - 모델 하이퍼파라미터를 자동으로 최적화하는 방법
# - 그리드(grid) 형태로 후보 값들을 모두 조합 → 교차검증으로 평가
# - 예시: max_depth, min_samples_split, criterion 등
# - 출력: 가장 좋은 성능을 내는 하이퍼파라미터 선택

# 🔹 요약
# - 과적합은 트리가 너무 복잡해서 발생
# - K-fold: 모델 성능을 안정적으로 평가
# - GridSearchCV: 하이퍼파라미터 최적화로 일반화 성능 향상
# - 두 가지를 함께 사용하면 '트리 과적합 방지 + 성능 향상' 가능

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris.keys())

train_data = iris.data
train_label = iris.target
print(train_data[:3], '\n',train_label[:3])

# 분류모델 생성
dt_clf = DecisionTreeClassifier()
dt_clf.fit(train_data, train_label)     # 모든 데이터를 학습에 사용하는 경우
pred = dt_clf.predict(train_data)       # 학습한 데이터로 예측 진행
print('예측값 : ', pred)
print('실제값 : ', train_label)
print('분류 정확도 : ', accuracy_score(train_label, pred))
# 분류 정확도 :  1.0
# 불순물이 있는 iris 데이터지만 1.0이 나와 과적합이 의심스러움.

# 과적합 방지 목적의 처리 1 : 학습 데이터, 검증 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=12)
dt_clf.fit(x_train, y_train)            # 일부 데이터만 학습에 사용하는 경우 (70%)
pred2 = dt_clf.predict(x_test)          # 일부 데이터만 예측 진행 (30%)
print('예측값 : ', pred2)
print('실제값 : ', y_test)
print('분류 정확도 : ', accuracy_score(y_test, pred2))
# 분류 정확도 :  0.9555555555555556
# 효과 : 앞선 모든 데이터를 사용해 학습하는 경우 과적합된 결과 였음을 확인

# 과적합 방지 목적의 처리 2 : K-fold 교차검증
# train_data를 분할해 학습과 평가를 병행하는 방법 중 하나
from sklearn.model_selection import KFold
import numpy as np
features = iris.data
label = iris.target
dt_clf2 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=12)
kfold = KFold(n_splits=5)       # k=5회
cv_acc = []
print('iris shape : ', features.shape)      # (150,4)
# KFold 학습시 전체 150행이 학습데이터(4/5, 120개) 검증 데이터(1/5 30개)로 분할되어 학습함.
n_iter = 0
# KFold 객체의 split()을 호출하면 Fold 별 학습용, 검증용 테스트의 행index를 array로 변환
for train_index, test_index in kfold.split(features):
    # print('n_iter(반복수) : ', n_iter)
    # print('train_index : ', train_index)
    # print('test_index : ', test_index)
    # n_iter += 1
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = label[train_index], label[test_index]
    
    # 학습 및 예측
    dt_clf2.fit(xtrain, ytrain)     # 80% 데이터로 학습 (Pareto 법칙)
    pred = dt_clf2.predict(xtest)   # 20% 데이터로 예측
    n_iter += 1

    # 반복 시행 마다 정확도 출력
    acc = np.round(accuracy_score(ytest,pred),5)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]

    print(f'반복수:{n_iter}, 교차검증 정확도:{acc}, 학습데이터크기:{train_size}, 검증크기:{test_size}')
    print(f'반복수:{n_iter}, 검증데이터 인덱스:{test_index}')
    cv_acc.append(acc)

print('cv_acc : ', np.array(cv_acc).astype(float))
print('평균 검증 정확도: ', np.mean(cv_acc))

# 참고 : StratifiedKFold - 불군형 분포를 가진 레이블 데이터 집합을 처리하기 위한 KFold

# 과적합 방지 목적의 처리 2-1 : 교차 검증 단순화
# cross_val_score를 이용해 교차 검증을 간단히 처리 가능
from sklearn.model_selection import cross_val_score     # 내부적으로 처리함
data = iris.data
label = iris.target

score = cross_val_score(dt_clf2, data, label, scoring='accuracy', cv=5)
print('교차 검증 별 정확도 : ', np.round(score, 3))
print('평균 검증 정확도: ', np.round(np.mean(score),3))

# 과적합 방지 목적의 처리 3 : GridSearchCV
# 간접적 과적합 방지 방법 : 최적의 파라미터 찾기(내부적으로 KFold 사용하기에 과적합을 줄이는데 도움을 줌)
from sklearn.model_selection import GridSearchCV
# 연습용으로 일부 파라미터만 사용 : max_depth, min_samples_split(node 분할을 위한 최소한의 샘플수로 과적합 제어)
parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]}
grid_dtree = GridSearchCV(estimator=dt_clf2, param_grid=parameters, cv=3, refit=True)
grid_dtree.fit(x_train, y_train)        # 내부적으로 복수 개의 모형을 생성하고 이를 실행시켜 최적의 parameter를 찾음

import pandas as pd
pd.set_option('display.max_columns', None)
score_df = pd.DataFrame(grid_dtree.cv_results_)
print(score_df)
print('GridSearchCV hyper parameter : ', grid_dtree.best_params_)
# GridSearchCV hyper parameter :  {'max_depth': 3, 'min_samples_split': 2}
print('GridSearchCV best score : ', grid_dtree.best_score_)
# GridSearchCV best score :  0.9238095238095237

# 최적의 모델
best_model = grid_dtree.best_estimator_     # 최적의 파라미터로 만들어진 모델
print(best_model)
best_pred = best_model.predict(x_test)
print('예측결과 : ', best_pred)
print('정확도 : ', accuracy_score(y_test,best_pred))

# 과적합 방지 기타 : 불필요한 변수 제거, 정규화(L1, L2), 데이터 양 증가, early stop...