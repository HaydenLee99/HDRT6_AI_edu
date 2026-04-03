from flask import Flask, render_template, request
import pymysql
import pandas as pd
import numpy as np
from markupsafe import escape   # HTML 특수문자를 변환해서 XSS 같은 공격을 막는 함수

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

@app.get("/dbshow")
def dbshow():
    dept = request.args.get("dept","").strip()

    sql="""
        select 
            j.jikwonno as 사번, j.jikwonname as 직원명, j.jikwonpay as 연봉, j.jikwonjik as 직급,
            b.busername as 부서명, b.busertel as 부서전화
        from jikwon j
        inner join buser b on j.busernum=b.buserno
    """

    params = []
    if dept:
        sql += " where b.busername like %s"
        params.append(f"%{dept}%")

    sql += " order by j.jikwonno asc"

    # SQL 실행
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]      # description : 컬럼 등 정보 얻기
    
    df = pd.DataFrame(rows, columns=cols)
    # print(df.head(3))

    # 직원 정보 html로 전송
    if not df.empty:
        jik_data = df[['사번','직원명','부서명','부서전화','연봉']].to_html(index=False)
    else:
        jik_data = "직원 정보가 없습니다."
    # print(jik_data.to_html(index=False))
    
    # 직급별 연봉 통계 자료
    if not df.empty:
        stats_data = (
            df.groupby('직급')['연봉']
            .agg(
                평균 = 'mean',
                표준편차 = lambda x:x.std(ddof=0),
                인원수 = 'count'
            )
            .round(2)
            .reset_index()      # index로 사용한 컬럼을 원래대로 되돌린다.
            .sort_values(by='평균', ascending=False)
        )
        stats_data['표준편차'] = stats_data['표준편차'].fillna(0)
        stats_data = stats_data.to_html(index=False)
        # print(stats_data)
    else:
        stats_data = "통계 대상 자료가 없습니다"

    return render_template('dbshow.html',
        dept=escape(dept),  # XSS 공격 방지
        jik_data=jik_data,
        stats_data=stats_data)


if __name__ == "__main__":
    app.run(debug=True)