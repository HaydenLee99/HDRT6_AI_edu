# pandas 문제 8)
# MariaDB에 저장된 jikwon, buser 테이블을 이용하여 아래의 문제에 답하시오.
# Django(Flask) 모듈을 사용하여 결과를 클라이언트 브라우저로 출력하시오.

#    1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
#        : 부서번호, 직원명 순으로 오름 차순 정렬 
#    2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.
#    3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.
#    4) 성별, 직급별 빈도표를 출력하시오.

from flask import Flask, render_template
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

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

@app.get("/")
def index():
    return render_template('index.html')

@app.get("/dbshow")
def dbshow():
    sql="""
        select 
            j.jikwonno as 사번, j.jikwonname as 직원명, b.busername as 부서명,
            j.jikwonjik as 직급, j.jikwonpay as 연봉, j.jikwongen as 성별,
            year(now()) - year(j.jikwonibsail) as 근무년수
        from jikwon j
        inner join buser b on j.busernum=b.buserno
        order by j.busernum, j.jikwonname
    """

    # SQL 실행
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]      # description : 컬럼 등 정보 얻기
    
    df = pd.DataFrame(rows, columns=cols)
    # print(df.head(3))

    if not df.empty:
        all_data = df[['사번', '직원명', '부서명', '직급', '연봉', '근무년수']].to_html(index=False)
        df_gj_crosstab = pd.crosstab(df['성별'], df['직급']).to_html()
        
        df_bj = df[['부서명', '직급', '연봉']]
        df_b = df_bj.pivot_table(values='연봉', index='부서명', aggfunc=['sum','mean']).round(2).reset_index()
        df_b.columns = ['부서명', '연봉합', '연봉평균']

        df_j = df_bj.pivot_table(values='연봉', index='직급', aggfunc=['sum','mean']).round(2).reset_index()
        df_j.columns = ['직급', '연봉합', '연봉평균']

        if not df_bj.empty:
            b_data = df_b.to_html(index=False)
            j_data = df_j.to_html(index=False)

            plt.figure()
            plt.bar(df_b['부서명'], df_b['연봉합'])
            plt.title('부서별 연봉합 그래프')
            plt.xlabel('부서명')
            plt.ylabel('연봉 합계[원]')
            plt.savefig('static/images/부서별 연봉합 그래프.png', bbox_inches='tight')
            plt.close()
        else:
            b_data = "부서별 연봉 정보가 없습니다."
            j_data = "직급별 연봉 정보가 없습니다."

    else:
        all_data = "전체 정보가 없습니다."
        df_gj_crosstab = "전체 정보가 없습니다."
        b_data = "전체 정보가 없습니다."
        j_data = "전체 정보가 없습니다."

    return render_template('dbshow.html',
        all_data=all_data,
        b_data=b_data, j_data=j_data,
        df_gj_crosstab=df_gj_crosstab)

if __name__ == "__main__":
    app.run(debug=True)