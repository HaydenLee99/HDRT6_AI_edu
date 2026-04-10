# [로지스틱 분류분석 문제2] - sklearn 버전
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 데이터 불러오기
data = pd.read_csv(
    "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/bodycheck.csv",
    usecols=['게임', 'TV시청', '안경유무']
)

# 독립변수, 종속변수
x = data[['게임', 'TV시청']]
y = data['안경유무']

# 학습용(train), 검증용(test) 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# 모델 생성 (sklearn은 기본적으로 규제 있음, 안정적)
model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)

# 예측
pred = model.predict(x_test)
print('예측값 :', pred)
print('실제값 :', y_test.values)

# 분류 정확도
acc = accuracy_score(y_test, pred)
print('분류 정확도 :', round(acc, 2))

# 모델 저장 및 로드
joblib.dump(model, 'logi_prac2_sklearn.model')
del model

read_model = joblib.load('logi_prac2_sklearn.model')

# 키보드 입력 예측
n = int(input('몇 개의 데이터를 입력하시겠습니까? '))
game = []
tv = []

for i in range(n):
    g = int(input(f'{i+1}번째 game 시간 입력 (0-23): '))
    t = int(input(f'{i+1}번째 TV 시청 시간 입력 (0-23): '))
    game.append(g)
    tv.append(t)

input_df = pd.DataFrame({'게임': game, 'TV시청': tv})
pred_input = read_model.predict(input_df)
print('예측값 :', pred_input)