import requests ##导入requests
from bs4 import BeautifulSoup ##导入bs4中的BeautifulSoup
from selenium import webdriver
import os
import time
import sqlite3


headers = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"}##浏览器请求头（大部分网站没有这个请求头会报错、请务必加上哦）
all_url = 'http://www.polyvore.com/cgi/search.sets?query=+&.search_src=masthead_search'  ##开始的URL地址

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches",["ignore-certificate-errors"])
driver = webdriver.Chrome(chrome_options=options)

driver.get(all_url)
driver.maximize_window()

i = 0

while i < 350:
#driver.execute_script("window.scrollBy(0,10000)")
 js = "window.scrollTo(0,document.body.scrollHeight)"
 driver.execute_script(js)
 time.sleep(1)
 i = i + 1
 print (i)

start_html = driver.page_source

driver.quit()

f = open('E:/polyvore7/html3months.txt','w',encoding='utf-8')
f.write(start_html)
f.close()



f = open('E:/polyvore7/html3months.txt','r',encoding='utf-8')
start_html = f.read()
print(start_html)
f.close()



