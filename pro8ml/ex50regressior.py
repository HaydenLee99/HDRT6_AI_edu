# sklearn 제공 Regressior 성능 비교
# pipeline + GridSearchCV + 교차검증 + 성능확인 + 시각화
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_diabetes      # 당뇨병 진행 정도 회귀 데이터
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

data = load_diabetes()                
# feature       의미      
# age           나이      
# sex           성별      
# bmi           체질량지수   
# bp            평균 혈압   
# s1            혈청 측정값 1
# s2            혈청 측정값 2
# s3            혈청 측정값 3
# s4            혈청 측정값 4
# s5            혈청 측정값 5
# s6            혈청 측정값 6

# label       의미 (당뇨 진행도)  
# 50          상태 양호     0
# 100         중간 정도     1
# 200+        상태 악화     2
# 300+        매우 심각     3

x = data.data
y = data.target
print("feature_names:", data.feature_names)
print("first row X:", x[:1])
print("target:", y[:5])

# 학습(80%), 검증(20%) 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Pipeline
models = {
    'LinearRegression':{
        'pipeline':Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'params':{
            'model__fit_intercept':[True, False]
        }
    },
    'RandomForest':{
        'pipeline':Pipeline([
            ('model', RandomForestRegressor(random_state=42))
        ]),
        'params':{
            'model__n_estimators':[100, 200],
            'model__max_depth':[None, 5, 10],
            'model__min_samples_split':[2, 5]
        }
    },
    'XGBoost':{
        'pipeline':Pipeline([
            ('model', XGBRegressor(random_state=42, verbosity=0))
        ]),
        'params':{
            'model__n_estimators':[100, 200],
            'model__learning_rate':[0.01, 0.05],
            'model__max_depth':[3, 5]
        }
    },
    'SVR':{
        'pipeline':Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR())
        ]),
        'params':{
            'model__C':[0.1, 1, 10],
            'model__gamma':['scale', 'auto'],
            'model__kernel':['rbf']
        }
    },
    'KNN':{
        'pipeline':Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsRegressor())
        ]),
        'params':{
            'model__n_neighbors':[3, 5, 7],
            'model__weights':['uniform', 'distance']
        }
    }
}

# GridSearchCV
results = []
best_models = {}

for name, config in models.items():
    print(f'{name} tunning...')
    grid = GridSearchCV(
        config['pipeline'],
        config['params'],
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    best_models[name] = best_model

    pred = best_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results.append([name, rmse, r2])

    print('best params : ', grid.best_params_)
    print('r2_score : ', r2)

# 최종 결과를 df로 저장
df_results = pd.DataFrame(results, columns=['model_name', 'RMSE', 'R2_score'])
df_results = df_results.sort_values('R2_score', ascending=False)
print("최종 성능 비교")
print(df_results)
# 최종 성능 비교
#       model_name       RMSE  R2_score
#              SVR  51.791775  0.493713
#          XGBoost  53.293563  0.463926
#     RandomForest  53.482640  0.460115
# LinearRegression  53.853446  0.452603
#              KNN  54.244609  0.444622

# 성능 비교를 위한 시각화
plt.figure(figsize=(12,5))

# R2_score 
plt.subplot(1,2,1)
sns.barplot(x="model_name", y="R2_score", data=df_results)
plt.title("튜닝 모델의 결정계수")
plt.xticks(rotation=30)

# RMSE
plt.subplot(1,2,2)
sns.barplot(x="model_name", y="RMSE", data=df_results)
plt.title("튜닝 모델의 RMSE")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# best model 예측 시각화
best_model_name = df_results.iloc[0]["model_name"]
best_model = best_models[best_model_name]
pred = best_model.predict(x_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')     # 100% 정확도 예측선
plt.title(f'최고 성능 모델 : {best_model_name}')
plt.xlabel("실제값")
plt.ylabel("예측값")
plt.show()