# 회귀분석 문제 4) 
# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
# Django 또는 Flask로 작성한 웹에서 근무년수를 입력하면 예상 연봉이 나올 수 있도록 프로그래밍 하시오.
# LinearRegression 사용. Ajax 처리!!!      참고: Ajax 처리가 힘들면 그냥 submit()을 해도 됩니다.

from flask import Flask, render_template, request
import pymysql
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

db_config = {
    'host':'localhost',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8mb4'
}

def get_connection():
    return pymysql.connect(**db_config)

@app.route("/")
def index():
    return render_template('index.html')

@app.get("/pred")
def pred():
    year = request.args.get('year', type=int)
    sql="""
        select 
            jikwonno as 사번, jikwonjik as 직급, jikwonpay as 연봉, year(now()) - year(jikwonibsail) as 근무년수
        from jikwon 
    """

    # SQL 실행
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
    
    df = pd.DataFrame(rows, columns=cols)
    # print(df.head(3))

    if not df.empty:
        df_jik = df[['직급', '연봉']]
        df_jik = df_jik.pivot_table(values='연봉', index='직급', aggfunc=['mean']).astype(int).reset_index()
        df_jik.columns = ['직급', '평균 연봉']
        j_data = df_jik.to_html(index=False)

    # 근무년수가 연봉에 영향을 주는 인과관계
    X = df[['근무년수']].values   # 2차원
    # print(X[:5])
    y = df['연봉'].values        # 1차원
    # print(y[:5])

    # 모델 생성
    model = LinearRegression().fit(X,y)

    # 회귀식
    slope = round(model.coef_[0],4)
    intercept = round(model.intercept_,4)

    # 결정계수 (%)
    y_pred = model.predict(X)
    r2 = round(r2_score(y, y_pred) * 100, 2)

    pred = None
    if year is not None:
        # 사용자가 입력한 값 예측
        pred = round(model.predict([[year]])[0])

    return render_template('pred.html',
        j_data=j_data, 
        slope=slope,
        intercept=intercept,
        r2=r2,
        pred=pred)

if __name__ == "__main__":
    app.run(debug=True)