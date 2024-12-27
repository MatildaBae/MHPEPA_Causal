#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.api import OLS, add_constant


# In[6]:


data = pd.read_csv("data/fixed_data.csv")
data


# In[8]:


# 결과 변수 목록
numeric_outcomes = ["AUN_SUM", "HEALTH", "K6SCMON"]

# 결과 저장
results = {}

# 수치형 결과 변수 처리
for outcome in numeric_outcomes:
    print(f"Estimating effect for numeric outcome (DiD + Parametric): {outcome}")

    # 변수 설정 (전체 데이터 사용)
    Y = data[outcome]
    T = data["Treatment"]
    Post = data["Post"]  # DiD의 시간 변수
    X = data.drop(columns=["Treatment", "Post"] + numeric_outcomes)

    # 성향 점수 계산 (Propensity Score)
    prop_model = LogisticRegression()
    prop_model.fit(X, T)
    propensity_scores = prop_model.predict_proba(X)[:, 1]

    # 처치와 성향 점수 추가
    X_with_ps = X.copy()
    X_with_ps['Propensity'] = propensity_scores
    X_with_ps['Post'] = Post
    X_with_ps['Treatment_Post'] = T * Post

    # 회귀 분석 (OLS)
    X_ols = add_constant(X_with_ps[['Propensity', 'Post', 'Treatment_Post']])
    ols_model = OLS(Y, X_ols).fit()

    # ATE 추정
    ate = ols_model.params['Treatment_Post']
    ci = ols_model.conf_int().loc['Treatment_Post']

    # 결과 저장
    results[outcome] = {
        "ATE": ate,
        "CI": (ci[0], ci[1])
    }

    print(f"ATE for {outcome}: {ate}, CI: ({ci[0]}, {ci[1]})\\n")

# 최종 결과 출력
for outcome, result in results.items():
    print(f"Outcome: {outcome}, ATE: {result['ATE']}, CI: {result['CI']}")


# In[9]:


import matplotlib.pyplot as plt

# 시각화 데이터 준비
outcomes = list(results.keys())
ates = [results[outcome]["ATE"] for outcome in outcomes]
ci_lowers = [results[outcome]["CI"][0] for outcome in outcomes]
ci_uppers = [results[outcome]["CI"][1] for outcome in outcomes]

# 막대 그래프 생성
plt.figure(figsize=(8, 6))
plt.bar(outcomes, ates, yerr=[np.array(ates) - np.array(ci_lowers), np.array(ci_uppers) - np.array(ates)],
        capsize=5, color=['blue', 'green', 'red'], alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Outcomes")
plt.ylabel("ATE")
plt.title("Estimated ATE with Confidence Intervals")
plt.show()


# In[ ]:




