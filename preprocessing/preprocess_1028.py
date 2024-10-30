#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[12]:


# ASC 파일 경로
file_path = 'LLCP2023.ASC '

# 파일을 numpy 배열로 읽기
brfss_2023 = pd.read_csv(file_path, sep='\t', header=None)

# 데이터 확인
brfss_2023


# In[20]:


brfss_2023.iloc(0)


# 2-state
# 
# 2-FMONTH(file month)
# 2-IMONTH(interview month)
# 2-IDAY(interview day)
# 4-IYEAR(interview year)
# 
# 4-DISPCODE(final disposition)
# 
