# bmi 지수 = 체중(kg) / 키(mm) / 키(mm)

# bmi.csv 만들기
# import random
# random.seed(12)

# def calc_bmiFunc(h, w):
#     bmi = w / (h/100) / (h/100)
#     if bmi < 18.5: return 'thin'
#     if bmi < 25.0: return 'normal'
#     return 'fat'

# fp = open('bmi.csv', mode='w')
# fp.write('height,weight,label\n')

# # 무작위 데이터 생성
# cnt = {'thin':0, 'normal':0, 'fat':0}

# for i in range(50000):
#     h = random.randint(150,200)
#     w = random.randint(35,100)
#     label = calc_bmiFunc(h,w)
#     cnt[label] += 1
#     fp.write('{0},{1},{2}\n'.format(h,w,label))
# fp.close()

# bmi.csv 읽기
import pandas as pd
df = pd.read_csv('bmi.csv')
print(df.shape)

# 최대값으로 나눠 0~1로 정규화
w = df['weight'] / 100
h = df['height'] / 200
wh = pd.concat([w,h], axis=1)
print(wh.head())

# label은 dummy화 : 범주형을 숫자형으로 변환
# thin 0 , normal 1 , fat 2
label = df['label'].map({'thin':0,'normal':1,'fat':2})
print(label.head())

# 학습, 검증 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(wh,label,test_size=0.3, random_state=1)

# SVM 생성 및 학습
from sklearn.svm import SVC
model = SVC(C=0.01, kernel='rbf').fit(x_train, y_train)

# 예측
pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)

# 평가
from sklearn.metrics import accuracy_score
sc_score = accuracy_score(y_test, pred)
print('분류 정확도 : ', sc_score)

# 교차 검증 모델
from sklearn import model_selection
cross_validation = model_selection.cross_val_score(model, wh, label, cv=3)
print('각 3회 별 정확도 : ', cross_validation)
print('평균 정확도 : ', cross_validation.mean())

# 새로운 값으로 예측
new_data = pd.DataFrame({'weight':[66,88],'height':[188,160]})
new_data['weight'] = new_data['weight'] / 100
new_data['height'] = new_data['height'] / 200
new_pred = model.predict(new_data)
print('새로운 값 예측 결과 : ', new_pred)

# 시각화
import matplotlib.pyplot as plt
df2 = pd.read_csv('bmi.csv', index_col=2)
def scatterFunc(lbl, color):
    b = df2.loc[lbl]
    plt.scatter(b['weight'], b['height'], c=color, label=lbl)

scatterFunc('fat', 'red')
scatterFunc('normal', 'yellow')
scatterFunc('thin', 'blue')
plt.legend()
plt.show()