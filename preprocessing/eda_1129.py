#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


df = pd.read_csv('data/final_processed.csv')
df


# In[17]:


df.info()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt


# x축 및 변수 이름
x_col = 'YEAR'
y_cols = ['AUN_SUM', 'AUPOPAMT', 'K6SCMON', 'HEALTH']

# 그래프 그리기
for y_col in y_cols:
    plt.figure(figsize=(10, 6))
    
    # NaN 제거
    valid_data = df[[x_col, y_col]].dropna()
    
    # 그룹별 평균 계산
    mean_data = valid_data.groupby(x_col).mean()
    
    # 선 그래프 그리기 (추세선)
    plt.plot(mean_data.index, mean_data[y_col], marker='o', label='Trend (Mean)')
    
    # 산점도 (간단히 추세 보조용)
    plt.scatter(valid_data[x_col], valid_data[y_col], alpha=0.3, label='Data Points', s=10)
    
    plt.title(f'{y_col} by {x_col}', fontsize=16)
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()


# In[16]:


df['HEALTH'].value_counts()


# In[19]:


# NaN 제거
valid_data = df[['YEAR', 'HEALTH']].dropna()

# 그룹별 평균 계산
mean_data = valid_data.groupby('YEAR').mean()

# 그래프 그리기
plt.figure(figsize=(10, 6))

# 선 그래프 (추세선)
plt.plot(mean_data.index, mean_data['HEALTH'], marker='o', label='Trend (Mean)')

# 산점도
plt.scatter(valid_data['YEAR'], valid_data['HEALTH'], alpha=0.3, label='Data Points', s=10)

# y축 제한
plt.ylim(0, 6)

# 그래프 설정
plt.title('HEALTH by YEAR', fontsize=16)
plt.xlabel('YEAR', fontsize=14)
plt.ylabel('HEALTH', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




