# =========================
# SVR + Optuna (TPE) + CV + TEST EVAL
# =========================

import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

import optuna

# =========================
# 1. Load data
# =========================
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")

# features = ['풍속(m/s)','비행고도(m)','2D이동거리(m)', '총회전량(deg)', '시뮬레이션전체시간(s)']
features = ['풍속(m/s)','비행고도(m)','2D이동거리(m)', '총회전량(deg)']

X = df_train[features]
y = df_train["배터리소모율(%)"]

X_test = df_test[features]
y_test = df_test["배터리소모율(%)"]

# =========================
# 2. Objective
# =========================
def objective(trial):

    C = trial.suggest_float("C", 1, 300, log=True)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.3)

    model = make_pipeline(
        StandardScaler(),
        SVR(
            kernel="rbf",
            C=C,
            epsilon=epsilon,
            gamma="scale"
        )
    )

    score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    ).mean()

    return -score

# =========================
# 3. Optuna Study
# =========================
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)

study.optimize(objective, n_trials=50)

# =========================
# 4. Train FINAL MODEL
# =========================
best_params = study.best_params

best_model = make_pipeline(
    StandardScaler(),
    SVR(
        kernel="rbf",
        C=best_params["C"],
        epsilon=best_params["epsilon"],
        gamma="scale"
    )
)
best_model.fit(X, y)

# =========================
# 5. TEST EVALUATION (ONCE ONLY)
# =========================
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== TEST RESULT ==========")
print("MAE:", mae)
print("R2 :", r2)

# =========================
# 6. SAVE MODEL
# =========================
joblib.dump(best_model, "best_svr_model.pkl")
joblib.dump(best_params, "best_svr_model_params.pkl")

print("\nSaved model and params.")
print("Best CV MAE:", study.best_value)
print("Best params:", best_params)

# CV vs TEST 비교
# CV 좋고 test 나쁘면 → 과적합
# 둘 비슷하면 → 안정 모델