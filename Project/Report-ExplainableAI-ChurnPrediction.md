# Explainable AI - Churn Prediction

[TOC]

## Introduction

Explainable AI (XAI) is crucial for understanding and trusting the decision-making process of complex machine learning models. In the context of churn prediction, where identifying customers likely to terminate their relationship with a product or service is paramount, ensemble algorithms have demonstrated high accuracy but suffer from reduced interpretability.

This project focuses on improving the explainability of these black-box models applied to a churn dataset taken from Kaggle. By utilizing LIME, Shapley values, and counterfactual explanations, we aim to provide insights into the factors driving churn predictions. These techniques shed light on feature contributions, individual instance explanations, and actionable counterfactual scenarios.

The report includes the dataset summary, model building and training with several competing algorithms such as lightgbm, catboost, XGboost, neural networks etc., an explanation of the XAI techniques, details of the experimental setup, the results and interpretations derived from applying XAI techniques, and a front-end on gradio to host the entire setup as a service. 

In conclusion, this project explores the growing field of explainable AI by using LIME, Shapley values, and counterfactual algorithms to provide transparent explanations for ensemble methods and neural nets in churn prediction. Our aim is to understand different techniques of explaining the black box of complex models, and discover how trust can be fostered in AI-driven decision-making processes.

## Data Summary







## Experimental Setup

### EDA

### Data Preprocessing

### Hyperparameter Optimization

### Model Results



## XAI Methods

### LIME

#### Overview

#### Results

### SHAP

#### Overview

#### Results

### Counterfactual Scenarios

#### Overview

#### Results



## Gradio

### Interface - LIME

### Interface - SHAP

### Interface - Counterfactual



## Conclusion