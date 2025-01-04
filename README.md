# MHPAEA Causal Inference Study 🧠📊

## Overview 🌟
This project examines the causal effects of the **Mental Health Parity and Addiction Equity Act (MHPAEA)** on the utilization of mental health services and overall mental well-being. Enacted in 2008, the MHPAEA requires parity in treatment between mental health/substance use disorder (MH/SUD) benefits and medical/surgical benefits under group health plans and insurance coverage. The primary aim is to measure the **average treatment effect (ATE)** for individuals with private insurance (treatment group) compared to those without private insurance (control group).

Additionally, the project evaluates **heterogeneous treatment effects (HTE)** by income level to explore policy impacts across socioeconomic groups.

## Objective 🎯
- **Primary Goals**:
  - Estimate the **causal effects** of MHPAEA on mental health service utilization and outcomes.
  - Assess the **heterogeneous treatment effects (HTE)** to determine whether the policy benefits vary across income groups.
- **Methodology**:
  - **Difference-in-Differences (DiD)**: To analyze pre- and post-policy effects.
  - **Doubly Robust Estimator**: To ensure robust ATE estimates by combining propensity score weighting with outcome regression.
  - **Subgroup Analysis**: Evaluate HTE by stratifying income groups.

## Data 📑
- **Source**: National Survey on Drug Use and Health (NSDUH), collected by SAMHSA (2005–2020).
- **Treatment Group**: Individuals with private health insurance eligible for MHPAEA.
- **Control Group**: Individuals without private health insurance.
- **Key Variables**:
  - **Treatment**: Binary indicator for MHPAEA applicability (private insurance status).
  - **Outcomes**:
    - Annual healthcare facility visits.
    - Self-reported health status (1-5, higher is better).
    - Mental health index (0-24, lower is better).
  - **Covariates**: Age, gender, education level, marital status, household size, income group, etc.

## Approach 🔍
### 1. **Data Preprocessing** 🧹:
- Cleaned and merged datasets to include treatment, outcomes, and covariates.
- Standardized continuous variables and encoded categorical variables as needed.

### 2. **Propensity Score Matching (PSM)** 🔄:
- Estimated propensity scores using logistic regression.
- Matched individuals from treatment and control groups to balance covariates.

### 3. **Difference-in-Differences (DiD)** 📉:
- Applied DiD to estimate the ATE by comparing pre- and post-policy differences between groups.

### 4. **Doubly Robust Estimator** 🏋️‍♂️:
- Combined propensity score weighting with outcome regression to estimate ATE.
- Used this method to improve robustness against model misspecification.

### 5. **Heterogeneous Treatment Effects (HTE)** 📊:
- Stratified the sample by income levels to compare treatment effects across socioeconomic groups.

## Results 📊
- **Overall Effects** (ATE):
  - **Annual healthcare visits**: Increased by 0.086.
  - **Self-reported health status**: Improved by 0.107.
  - **Mental health index**: Improved by 0.182.
- **HTE by Income Level**:
  - The most significant improvements were observed in lower-income groups, suggesting the policy had the greatest impact on underserved populations.
  - Middle-income groups experienced relatively weaker effects, highlighting potential areas for policy refinement.

## Policy Implications 💡
- MHPAEA effectively improved mental health service utilization and outcomes, particularly for lower-income groups.
- Future policies should address gaps in effectiveness among middle-income groups to maximize overall societal benefit.

## Limitations ⚙️
- Potential unobserved confounders may still bias results.
- Long-term effects require further investigation.

## Future Work 🚀
- Enhance robustness of subgroup analyses using advanced matching methods.
- Explore the policy’s impact on broader mental health outcomes, such as long-term economic productivity and quality of life.

---

# MHPAEA 인과추론 연구 🧠📊

## 개요 🌟
이 프로젝트는 **Mental Health Parity and Addiction Equity Act (MHPAEA)** 가 정신건강 서비스 이용과 전반적인 정신건강 상태에 미친 인과적 영향을 분석합니다. 2008년에 제정된 MHPAEA는 단체 건강보험과 의료/외과 혜택을 제공하는 보험에서 정신건강/약물 사용 장애(MH/SUD) 혜택 간의 평등을 보장하도록 요구합니다. 이 연구의 주요 목표는 민간 보험 가입자(치료 집단)와 비가입자(대조 집단) 간의 **평균 처리 효과(ATE)** 를 측정하는 것입니다.

또한, 소득 수준에 따른 **이질적 처리 효과(HTE)** 를 평가하여 정책의 사회경제적 영향을 탐구합니다.

## 목표 🎯
- **주요 목표**:
  - MHPAEA가 정신건강 서비스 이용 및 결과에 미친 **인과적 효과** 를 추정합니다.
  - 소득 계층에 따른 **이질적 처리 효과(HTE)** 를 평가하여 정책 혜택의 차이를 분석합니다.
- **방법론**:
  - **이중차분법(DiD)**: 정책 시행 전후의 효과를 분석합니다.
  - **Doubly Robust Estimator**: Propensity Score와 결과 회귀 모델을 결합해 더 강건한 ATE를 추정합니다.
  - **소그룹 분석**: 소득 계층별 HTE를 평가합니다.

## 데이터 📑
- **출처**: SAMHSA에서 수집한 약물 사용 및 건강에 관한 국가 조사(NSDUH) 데이터(2005–2020).
- **치료 집단**: MHPAEA가 적용되는 민간 건강보험 가입자.
- **대조 집단**: 민간 건강보험에 가입되지 않은 대상자.
- **주요 변수**:
  - **처치**: MHPAEA 적용 여부(민간 보험 가입 여부로 구분된 이진 변수).
  - **결과 변수**:
    - 연간 의료시설 방문 횟수.
    - 자기 보고 건강 상태(1-5, 높을수록 건강).
    - 정신건강 지수(0-24, 낮을수록 건강).
  - **공변량**: 연령, 성별, 교육 수준, 혼인 상태, 가구원 수, 소득 수준 등.

## 접근법 🔍
### 1. **데이터 전처리** 🧹:
- 처치, 결과 변수, 공변량을 포함하도록 데이터 정리 및 병합.
- 연속형 변수는 표준화하고 범주형 변수는 인코딩.

### 2. **Propensity Score Matching (PSM)** 🔄:
- 로지스틱 회귀를 사용해 Propensity Score를 추정.
- Propensity Score를 기반으로 치료 집단과 대조 집단을 매칭하여 공변량 균형을 맞춤.

### 3. **이중차분법(DiD)** 📉:
- 정책 시행 전후 두 집단 간 차이를 비교하여 ATE를 추정.

### 4. **Doubly Robust Estimator** 🏋️‍♂️:
- Propensity Score와 결과 회귀 모델을 결합하여 ATE를 추정.
- 모델 명세 오류에 강건한 추정을 위해 활용.

### 5. **이질적 처리 효과(HTE)** 📊:
- 소득 수준별로 데이터를 나누어 정책 효과를 비교.

## 결과 📊
- **전체 효과(ATE)**:
  - **연간 의료시설 방문 횟수**: 0.086 증가.
  - **자기 보고 건강 상태**: 0.107 개선.
  - **정신건강 지수**: 0.182 개선.
- **소득 계층별 HTE**:
  - 저소득층 그룹에서 가장 큰 개선 효과가 나타났으며, 정책이 소외 계층에 큰 영향을 미친 것으로 보임.
  - 중간 소득 그룹에서는 상대적으로 약한 효과가 관찰되어 정책 개선의 여지가 있음.

## 정책적 시사점 💡
- MHPAEA는 정신건강 서비스 이용과 결과를 개선하는 데 효과적이었으며, 특히 저소득층에서 두드러진 효과를 보였습니다.
- 향후 정책은 중간 소득 그룹에서의 효과를 강화하여 사회적 혜택을 극대화해야 합니다.

## 한계 ⚙️
- 잠재적 혼란 변수(Confounder)의 누락 가능성이 여전히 존재합니다.
- 장기적인 효과를 평가하기 위해 추가 연구가 필요합니다.

## 향후 과제 🚀
- 소그룹 분석의 강건성을 높이기 위해 고급 매칭 기법을 활용합니다.
- 정책이 경제적 생산성과 삶의 질과 같은 장기적인 정신건강 결과에 미친 영향을 탐구합니다.
