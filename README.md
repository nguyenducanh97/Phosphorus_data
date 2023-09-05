# Affiliation

This repository contains code for the paper entitled "Super-sorption Capacity of Phosphorus from aqueous solutions by Ultrahigh-capacity and Porosity Ranunculus-like MgO" authored by Duc Anh Nguyen, Viet Bac Nguyen, and Am Jang from the Department of Global Smart City and Electrical and Computer Engineering, Sungkyunkwan University (SKKU), 2066, Seobu-ro, Jangan-gu, Suwon-si, Gyeonggi-do 16419, Republic of Korea. The published article can be found in ACS Environmental Science and Technology.

## Introduction

Due to the rapid growth in the manufacturing industries, phosphorus (P) contamination has become an urgent wastewater issue. In this research, we demonstrate the exceptional potential of Ranunculus-like MgO calcined at 400-600°C (MO4-MO6) as superior adsorbents for phosphate removal and predict the adsorption performance using several machine learning models. A data frame with 318 rows and 19 columns was created from 954 experimental laboratory tests. In other words, each data point was averaged from triplicated results to avoid overfitting issues. The first 15 columns of the data frame represent input features, including initial pH, dosage of adsorbent, contact time, initial adsorbate concentration, temperature, and the concentration of individual and mixed co-existing components (NO3–, Cl–, HCO3–, SO42–, F–, humic acid, K+, Na+, NH4+, COD). The remaining four columns correspond to output features, namely adsorption capacity, removal efficiency, final pH, and leaching Mg. Eight traditional machine learning models (Multiple Linear Regression, ElasticNet Regression, Random Forest, Extra Trees, Lasso, Ridge, BaggingRegressor, KNeighborsRegressor), two deep neural network models (DNN ver 1 and DNN ver 2), and two Deep Belief Network models (DBN ver 1, DBN ver 2) were examined and chosen for further use.

## Files in this Repository

- `<dbn>`: Library for Deep Belief Network models.
- `<Data-for-P-MgO.csv>`: Dataset applied in this study.
- `<Model>`: Models after being trained using the prepared dataset.
- `<FeIm for DBN>` and `<PDP data for DBN ver 5>`: Feature importance and partial dependence plot data for DBN models, respectively.
- `<DBN regression for P-MgO.py>`: DBN and feature importance algorithms.
- `<Traditional Machine Learning for P-MgO.ipynb>`: DNN, traditional machine learning, and feature importance algorithms.
- `<PDP for DBN regression.py>`: The partial dependence plot (PDP) algorithms.

## Findings

A total of 10 machine learning (ML) models were applied to simultaneously predict multi-criteria (i.e., sorption capacity, removal efficiency, final pH, and Mg leakage) affected by 15 input features. Traditional ML models and Deep Neural Networks initially face challenges with poor accuracy, especially for removal efficiency. However, the breakthrough comes with applying Deep Belief Network (DBN) with unparalleled prediction performance (MAE = 1.3289, RMSE = 5.2555, R2 = 0.9926) across all output features, surpassing all current studies using 4 times higher number of data points for predicting only one output factor. This captivating MO6 and DBN model are believed to open immense horizons to transformative applications, especially for P removal in the near future.

## Correspondence

- Email: nguyenducanh@g.skku.edu
- Phone: +82-10-2816-9711 (Korea)

Thank you very much.
