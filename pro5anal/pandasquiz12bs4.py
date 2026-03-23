import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
import csv

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "Mozilla/5.0"}

def get_page_data(page):
    url = f"https://finance.naver.com/sise/sise_market_sum.naver?&page={page}"
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.content, 'lxml')

    # 컬럼은 첫 페이지에서만 가져오기
    if page == 1:
        cols = soup.select("thead tr th")
        cols = [c.get_text(strip=True) for c in cols[:-1]]
    else:
        cols = None
    
    all_td = soup.select("tbody tr td")
    all_td = [td.get_text(strip=True).replace(',', '') for td in all_td]
    
    clean = [i for i in all_td if i != '']
    
    result = []
    for i in range(0, len(clean), 12):
        chunk = clean[i:i+12]
        if len(chunk) == 12:
            result.append(chunk)
    
    return result, cols

data = []
END_PAGE = 2
for page in range(1, END_PAGE + 1):
    data.extend(get_page_data(page)[0])

df = pd.DataFrame(data, columns=get_page_data(page)[1])
print(df.head())

file_name = f"naver_finance_1to{50*END_PAGE}.csv"
df.to_csv(file_name, index=False, encoding='utf-8')

with open(file_name, mode='w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(df.columns)
    writer.writerows(df.values)

df2 = pd.read_csv(file_name, encoding='utf-8')
print(df2.head())
print(df2.info())