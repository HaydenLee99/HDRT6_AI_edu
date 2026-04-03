# pandas의 dataframe 자료를 원격 DB의 테이블에 저장
import pymysql
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import csv
import os

def main():
    print("\tmain을 실행합니다.\n")
    data={
        'code':[6,7,8],
        'sang':['사이다','선풍기','에어컨'],
        'su':[20,22,5],
        'dan':[1000,20000,1000000]
    }
    try:
        frame = pd.DataFrame(data)
        print(frame)
        engine = create_engine("mysql+pymysql://root:123@127.0.0.1:3306/test?charset=utf8")

        frame.to_sql(name="sangdata", con=engine, if_exists="replace", index=False)

        df = pd.read_sql("select code, sang, su, dan from sangdata", engine)
        print(df)

    except Exception as e:
        print(e)

    finally:
        engine.dispose()

    print("\tmain을 종료합니다.\n")

if __name__ == "__main__":
    main()