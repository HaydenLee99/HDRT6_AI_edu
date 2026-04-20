# MLP (Multi-Layer Perceptron) 핵심 정리
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_wine()
x, y = data.data, data.target
print(x.shape)          # (178, 13)
print(np.unique(y))     # [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42, stratify=y)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

model = MLPClassifier(
    max_iter=100,
    hidden_layer_sizes=(20,10),
    solver='adam',
    activation='relu',
    learning_rate_init=0.001,
    random_state=42,
    verbose=1
    )
model.fit(x_train_scaled, y_train)

pred = model.predict(x_test_scaled)
print('실제값 : ', y_test[:10])
print('예측값 : ', pred[:10])
print('acc : ', accuracy_score(y_test, pred))
print('classification_report:\n', classification_report(y_test, pred))
print('confusion_matrix:\n', confusion_matrix(y_test, pred))

# confusion_matrix 시각화
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('confusion_matrix')
plt.show()

# train loss curve 시각화
plt.plot(model.loss_curve_)
plt.title('train loss curve')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.grid(True)
plt.show()
# MLP는 미분으로 오차를 줄여 나간다.

