#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas pyreadstat


# In[9]:


import pandas as pd
import pyreadstat

# XPT 파일 읽기
file_path = 'LLCP2020.XPT '  # XPT 파일의 경로를 지정합니다.
df, meta = pyreadstat.read_xport(file_path)

# 데이터프레임 확인 (선택사항)
print(df.head())  # 데이터의 처음 몇 줄을 출력합니다.

# CSV 파일로 저장
csv_file_path = 'LLCP2020.csv'  # 저장할 CSV 파일 경로
df.to_csv(csv_file_path, index=False)  # 인덱스를 포함하지 않고 CSV로 저장


# In[6]:


len(df)


# In[ ]:




