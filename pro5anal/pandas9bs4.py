# BeautifulSoup 객체를 이용한 웹 문서 처리 연습
import requests
from bs4 import BeautifulSoup

baseurl = "https://www.naver.com"
headers = {"User-Agent":"Mozilla/5.0"}              # Mozilla/5.0 엔진을 사용하여 서버에 요청함

source = requests.get(baseurl, headers=headers)
print(source, type(source))                         # <Response [200]> <class 'requests.models.Response'>
print(source.status_code)                           # 200
# print(source.text, type(source.text))             # <!doctype html>... </html>  <class 'str'> String으로 넘어옴
# print(source.content)                             # binary data로 넘어옴
# 상기 데이터를 다룰 수 있도록 돕는 것이 BeautifulSoup

conv_data = BeautifulSoup(source.text, 'lxml')
# print(conv_data, type(conv_data))                 # <!doctype html>... </html>  <class 'bs4.BeautifulSoup'> bs객체로 넘어옴
for atag in conv_data.find_all('a'):                # .find_all('a') : 모든 a_tag를 가져온다.
    href = atag.get('href')
    title = atag.get_text(strip = True)             # .get_text(strip = True) 공백 제거 설정 킴
    if title:
        print(href, title)
        print('-' * 30)

# 크롤링 시 보안으로 막히면...
# 동적 웹크롤링 selenium을 이용하면 JS로 브라우저 통제가능해 크롤링이 가능해지긴 하지만 작업량이 많음
# 크롤링 방지를 위해 사이트 보안으로 관리자가 html 구조를 주기적으로 변경하는 경우가 많음. -> 크롤링 데이터 활용 어려움