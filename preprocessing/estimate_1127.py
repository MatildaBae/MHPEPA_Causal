#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('final_data.csv')
df


# In[4]:


df.columns


# In[3]:


df.info()


# In[10]:


df['IRHHSIZ2'].value_counts()


# In[6]:


df.info()


# In[ ]:


import pandas as pd

# 데이터 불러오기
# data = pd.read_csv("your_data.csv")  # CSV 파일 경로를 여기에 입력하세요.

# 1. 결측값 처리 (필요에 따라 수정)
data = data.dropna(subset=["K6SCMON"])  # K6SCMON이 결과 변수이므로 결측값 제거
data.fillna(0, inplace=True)  # 다른 결측값은 0으로 대체 (필요에 따라 조정 가능)

# 2. 처치 변수 생성 (정책 도입 전후 여부)
data["Treatment"] = (data["year_x"] >= 2010).astype(int)

# 3. 결과 변수와 공변량 지정
Y = data["K6SCMON"]  # 결과 변수
A = data["Treatment"]  # 처치 변수
X = data[["AGE2", "IRSEX", "EDUCCAT2", "INCOME", "SERVICE"]]  # 공변량 선택

# 소득 계층 나누기 (1: 저소득, 4: 고소득 기준으로 나눔)
data["IncomeGroup"] = pd.cut(data["INCOME"], bins=[0, 1, 2, 3, 4], labels=["Low", "Middle-Low", "Middle-High", "High"])


# ![image.png](attachment:image.png)

# In[11]:


df_jw = df[['AGE2', 'IRSEX', 'INCOME', 'IRMARIT', 'SERVICE',
           'CG30EST', 'AL30EST']]
df_jw


# In[12]:


import pandas as pd
import numpy as np


# 'AGE2', 'IRSEX', 'INCOME', 'IRMARIT', 'SERVICE' 명목형으로 처리
# 명목형은 기본적으로 category로 처리 가능
df_jw['AGE2'] = df_jw['AGE2'].astype('category')
df_jw['IRSEX'] = df_jw['IRSEX'].astype('category')
df_jw['INCOME'] = df_jw['INCOME'].astype('category')
df_jw['IRMARIT'] = df_jw['IRMARIT'].astype('category')

# 'SERVICE': 군 복무 여부, 85, 89, 97, 98 → NaN
df_jw['SERVICE'] = df_jw['SERVICE'].replace([85, 89, 97, 98], np.nan).astype('category')

# 'CG30EST': 근 1달간 담배 피운 일수 → 수치형 처리
df_jw['CG30EST'] = df_jw['CG30EST'].replace({91: 0, 93: 0, 99: 0, 94: np.nan, 97: np.nan, 98: np.nan}).astype(float)

# 'AL30EST': 근 1달간 알코올 마신 일수 → 수치형 처리
df_jw['AL30EST'] = df_jw['AL30EST'].replace({91: 0, 93: 0, 99: 0, 94: np.nan, 97: np.nan, 98: np.nan}).astype(float)

# 결과 출력
df_jw


# In[13]:


df_jw.info()


# In[20]:


df_jw['CG30EST'].value_counts()


# In[21]:


df_jw['CG30EST'].isna().sum()


# In[22]:


df_jw.to_csv('data/preprocessed_jiwon.csv')


# In[23]:


df_sh = pd.read_csv('data/preprocessed_soy.csv')
df_sh


# In[29]:


df = pd.read_csv('final_data.csv')
df = df[['QUESTID2', 'year_x']]
df


# In[30]:


df_jj = pd.read_csv('data/processed_jeongje.csv')
df_jj = df_jj[['TXEVER', 'TXYREVER', 'AUN_SUM', 'AUUN_ANY', 'HEALTH', 'K6SCMON']]
df_jj


# In[31]:


df_jj.columns


# In[32]:


fin = pd.concat([df, df_jw, df_sh, df_jj], axis=1)
fin


# In[33]:


fin.columns


# In[34]:


fin.to_csv('data/final_processed.csv')


# In[10]:


import pandas as pd

df_jj = pd.read_csv('data/processed_jeongje.csv')
df_jj = df_jj[['QUESTID2','AUPOPAMT', 'AUUNCOST', 'AUUNNCOV', 'AUUNENUF']]
df_jj


# In[11]:


df_jj.info()


# In[12]:


import numpy as np
df_jj['AUUNCOST'] = df_jj['AUUNCOST'].replace({6: 1, 3: 0, 85: np.nan, 94: np.nan, 97: np.nan, 98: np.nan, 99: np.nan}).astype(float)
df_jj['AUUNNCOV'] = df_jj['AUUNNCOV'].replace({6: 1, 3: 0, 85: np.nan, 94: np.nan, 97: np.nan, 98: np.nan, 99: np.nan}).astype(float)
df_jj['AUUNENUF'] = df_jj['AUUNENUF'].replace({6: 1, 3: 0, 85: np.nan, 94: np.nan, 97: np.nan, 98: np.nan, 99: np.nan}).astype(float)

df_jj['AUUNCOST'] = df_jj['AUUNCOST'].astype('category')
df_jj['AUUNNCOV'] = df_jj['AUUNNCOV'].astype('category')
df_jj['AUUNENUF'] = df_jj['AUUNENUF'].astype('category')


df_jj.info()


# In[13]:


df_jj


# In[9]:


fin = pd.read_csv('data/final_processed.csv')
fin


# In[14]:


# Merge on 'QUESTID2'
merged_df = pd.merge(df_jj, fin, on='QUESTID2', how='inner')  # Use 'left', 'right', or 'outer' if needed

merged_df


# In[15]:


merged_df.info()


# In[16]:


merged_df.to_csv('data/final_processed.csv', index=False)


# In[ ]:




