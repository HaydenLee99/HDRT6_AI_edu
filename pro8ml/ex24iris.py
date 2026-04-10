# 트리탐색 - iris
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# 이를 softmax regression 또는 multinormial logistic regression이라 부름

iris = datasets.load_iris()
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
print(iris.data[:3])
print(iris.target[:3])
print(np.corrcoef(iris.data[:,2], iris.data[:,3])[0,1])     # 0.9628654314027961

x = iris.data[:,[2,3]]
y = iris.target
print(x.shape,' ',y.shape)
print(x[:3])
print(y[:3], set(map(int, y)))

# 학습 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train[:3], '\n', x_test[:3], '\n', y_train[:3], '\n', y_test[:3])

# # feature scaling - 최적화 과정에서 안정성, 수렴속도 향상, 과적합/과소적합 방지 등의 효과 있음
# sc = StandardScaler()
# sc.fit(x_train)

# x_train = sc.transform(x_train)
# x_test = sc.transform(x_test)

# print(x_train[:3])
# print(x_test[:3])

# # 스케일링 결과 번복
# ori_x_train = sc.inverse_transform(x_train)
# print(ori_x_train[:3])
# iris data는 크기의 차이가 거의 없어 표준화의 의미가 없음

# 분류 모델 생성
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)
model.fit(x_train, y_train)
# 모델에 규제 적용 1 : C (L2규제) C가 작을수록 규제가 강함, 1.0(default)
# 숫자값을 조정해가며 정확도 확인
# 모델에 규제 적용 2 : solver='lbfgs' (softmax 지원)
# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred[:5])
print('실제값 : ', y_test[:5])

print(f'총개수 : {len(y_test)}, 오류수 : {(y_test != y_pred).sum()}')

# 분류 정확도 확인 1
print(f'분류 정확도 : {accuracy_score(y_test, y_pred)}')

# 분류 정확도 확인 2
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측치'], colnames=['관측치'])
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2])/len(y_test))

# 분류 정확도 확인 3
print('test score : ', model.score(x_test, y_test))     # 두 값의 차이가 크다면 과적합 의심 필요
print('train score : ', model.score(x_train, y_train))  # 두 값의 차이가 크다면 과적합 의심 필요

# 학습된 모델 저장 후 읽기
import joblib
joblib.dump(model, 'logmodel.pkl')
del model

read_model = joblib.load('logmodel.pkl')

# 이 후에는 read_model 사용
new_data = np.array([[5.5, 2.2],[0.6, 0.3],[1.1,0.5]])

# 주의, 표준화된 자료로 모델을 생성했다면 new_data도 반드시 표준화 해야 함
# sc.fit(new_data)
# new_data = sc.transform(new_data)

new_pred = read_model.predict(new_data)
print('예측 결과 : ',new_pred)
# 위의 결과는 softmax의 확률값 중 가장 큰 인덱스가 출력된 값이다. 
# predict()는 사실상 argmax(predict_proba) 한 것과 동일
print(read_model.predict_proba(new_data))   # softmax 실제 출력값

# 시각화
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')      # 마커 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)   # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)   
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regionFunc(X=x_combined_std, y=y_combined, classifier=read_model, test_idx=range(105, 150), title='scikit-learn제공')  