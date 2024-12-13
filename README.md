# MHPAEA Causal Inference Study 🧠📊

## Overview 🌟
This project investigates the causal effect of the **Mental Health Parity and Addiction Equity Act (MHPAEA)** on the utilization of mental health services and specialist consultations. The MHPAEA, enacted in 2008, mandates parity between mental health/substance use disorder (MH/SUD) benefits and medical/surgical benefits in group health plans and health insurance coverage. The focus of this study is to compare the **average treatment effect (ATE)** between individuals who received the treatment (those with private insurance) and those who did not. 

The study uses a **Difference-in-Differences (DID)** approach to estimate the causal effect, comparing pre- and post-treatment differences between the treatment and control groups. Additionally, the analysis adjusts for confounders using **propensity score matching**, ensuring more accurate comparison between groups.

## Objective 🎯
- **Primary Goal**: Estimate the causal effect of MHPAEA on the utilization of mental health services by comparing the treatment group (private insurance holders) with the control group (non-private insurance holders).
- **Methodology**: 
  - **Difference-in-Differences (DID)**: To compare the pre- and post-treatment effects between the treatment and control groups.
  - **Propensity Score Matching (PSM)**: To adjust for confounding variables and make the treatment and control groups comparable.

## Data 📑
- **Treatment Group**: Individuals with private insurance who are eligible for MHPAEA.
- **Control Group**: Individuals without private insurance who are not eligible for MHPAEA.
- **Variables**:
  - **Treatment**: Whether the individual has private insurance (treated) or not (control).
  - **Outcome**: Utilization of mental health services and specialist consultations.
  - **Covariates**: Age, gender, socioeconomic status, health conditions, etc.

## Approach 🔍
1. **Data Preprocessing** 🧹:
   - Clean and merge datasets containing information on insurance status, mental health service utilization, and relevant covariates.
   - Identify eligible individuals based on insurance status.

2. **Propensity Score Matching** 🔄:
   - Estimate propensity scores using logistic regression or other suitable methods.
   - Match individuals in the treatment and control groups based on the estimated propensity scores to reduce bias due to confounders.

3. **Difference-in-Differences (DID)** 📉:
   - Implement DID to estimate the effect of MHPAEA by comparing changes in utilization before and after the law’s implementation between the treatment and control groups.

4. **Statistical Analysis** 📊:
   - Perform robustness checks and sensitivity analysis to assess the validity of the causal estimates.

## Future Work 🚀
- Further refine the propensity score matching method.
- Explore other potential treatment effects, such as subgroup analysis based on demographics.
- Investigate the long-term impact of MHPAEA on mental health outcomes.

/--

# MHPAEA 인과추론 연구 🧠📊

## 개요 🌟
이 프로젝트는 **Mental Health Parity and Addiction Equity Act (MHPAEA)**가 정신건강 서비스와 전문 상담 이용에 미친 인과적 영향을 조사합니다. MHPAEA는 2008년에 제정된 법으로, 민간 건강 보험에서 정신건강/약물 중독 치료와 의료/외과 치료 간의 평등을 보장하도록 요구합니다. 이 연구의 초점은 **평균 처리 효과(ATE)**를 비교하는 것입니다. 즉, 민간 보험 가입자(치료 집단)와 비가입자(대조 집단) 간의 정신건강 서비스 및 전문 상담 이용 차이를 비교합니다.

이 연구는 **Difference-in-Differences (DID)** 접근법을 사용하여 인과적 효과를 추정하며, 치료 전후의 차이를 비교합니다. 또한 **propensity score matching**을 사용하여 혼동 변수를 조정하고 두 집단을 보다 정확하게 비교합니다.

## 목표 🎯
- **주요 목표**: MHPAEA가 정신건강 서비스 이용에 미친 인과적 효과를 추정하기 위해, 민간 보험 가입자(치료 집단)와 비가입자(대조 집단) 간의 차이를 비교합니다.
- **방법론**: 
  - **Difference-in-Differences (DID)**: 치료 전후의 변화를 비교하여 처리 효과를 추정합니다.
  - **Propensity Score Matching (PSM)**: 혼동 변수를 조정하여 치료 집단과 대조 집단을 비교 가능하게 만듭니다.

## 데이터 📑
- **치료 집단**: MHPAEA 적용 대상인 민간 보험 가입자.
- **대조 집단**: MHPAEA 적용을 받지 않는 비보험 가입자.
- **변수**:
  - **처치**: 민간 보험 가입 여부 (치료 집단: 처치, 대조 집단: 대조 집단).
  - **결과**: 정신건강 서비스 및 전문 상담 이용 여부.
  - **공변량**: 나이, 성별, 사회경제적 상태, 건강 상태 등.

## 접근법 🔍
1. **데이터 전처리** 🧹:
   - 보험 상태, 정신건강 서비스 이용 정보 및 관련 공변량을 포함한 데이터 정리 및 병합.
   - 보험 가입 여부에 따라 연구 대상자 선정.

2. **Propensity Score Matching** 🔄:
   - 로지스틱 회귀 등의 방법으로 propensity score 추정.
   - 추정된 propensity score를 바탕으로 치료 집단과 대조 집단을 매칭하여 혼동 변수의 영향을 최소화.

3. **Difference-in-Differences (DID)** 📉:
   - DID 방법을 적용하여 MHPAEA 법안 시행 전후의 이용 변화량을 비교하여 처리 효과 추정.

4. **통계 분석** 📊:
   - 인과 추정의 유효성을 검토하기 위한 강건성 검정 및 민감도 분석 수행.

## 향후 작업 🚀
- Propensity score matching 방법 개선.
- 인구통계학적 변수에 따른 하위 그룹 분석 등 다른 처리 효과 탐색.
- MHPAEA의 정신건강 결과에 대한 장기적 영향 분석.
