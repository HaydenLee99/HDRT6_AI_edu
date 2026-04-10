# ROC Curve

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

x,y = make_classification(n_samples=100, n_features=2, n_redundant=0),random_state=42

# 산포도
import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1])
# plt.show()

model = LogisticRegression().fit(x,y)
y_hat = model.predict(x)

# Roc curve의 판별경계썬 설정용 결정함수 사용
f_value = model.decision_function(x)
df = pd.DataFrame(np.vstack([f_value, y_hat,y]).T, columns=[f_value, y_hat,y])

# 모델 성능 파악
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y,y_hat))

from sklearn import metrics
fpr, tpr, thresholds = metrics. roc_curve(y,model.decision_function)