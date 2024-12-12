# MHPAEA Causal Inference Study 🧠📊

## Overview 🌟
This project investigates the causal effect of the **Mental Health Parity and Addiction Equity Act (MHPAEA)** on the utilization of mental health services and specialist consultations. The focus is on comparing the average treatment effect (ATE) between individuals who received the treatment (those with private insurance) and those who did not. The study uses a **Difference-in-Differences (DID)** approach to estimate the causal effect, adjusting for confounders using **propensity score matching**.

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
