# Data Science Fundamentals - Predicting Salary Based on LinkedIn Dataset

# 1. Introduction

This project is a part of the Data Science Fundamentals course at ISCTE-IUL. The goal of this project is to predict the salary of a LinkedIn user based on the information available on job offers. The dataset used in this project is the LinkedIn dataset, which is available on Kaggle. The dataset is available on Kaggle: [LinkedIn Dataset](https://www.kaggle.com/datasets/adampq/linkedin-jobs-machine-learning-data-set)

# 2. Project Structure

The project is based on the CRISP-DM methodology. The project structure is as follows:

- Business Understanding (Won't be covered in this project)
- Data Understanding
- Data Preparation
- Modeling
- Evaluation
- Deployment (Won't be covered in this project)

# 3. libraries

All libraries can be installed using pip. There is a requirements.txt file in the project folder. To install all libraries, run the following command in the project folder: `pip install -r requirements.txt`

# Data Understanding

The understanding of the dataset was done in `Understand&Prepare.ipynb` notebook and all the code and results are presented there. It includes statistics presentation, data visualization and correlation analysis.

# Data Preparation

The data preparation was done in `Understand&Prepare.ipynb` notebook and all the code and results are presented there. It includes data cleaning and feature selection.

# Modeling

For that purpose we will use the following algorithms:

- K Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- Gradient Boosting Regressor

With the feautres we were left with Sure, the features in your dataset appear to be:

1. `Flw_Cnt`
2. `Is_Supvsr`
3. `is_remote`
4. `views`
5. `mean_sal_by_st_code`
6. `mean_sal_by_xp_lvl`
7. `mean_sal_by_wrk_type`
8. `NLP - Job_Decscription`

All models are Regression models, because we are predicting a continuous variable. We could also use Classification models, but we would have to divide the salary into classes. In this case, we would have to decide how many classes we want to have and what would be the salary range for each class. This would be a very subjective decision and it would be hard to evaluate the results.

The modeling was done in `ModelTesting.ipynb` notebook and all the code and results are presented there. It includes data modeling, hyperparameter tuning and model evaluation. Great help was provided by GridSearchCV, which is a part of the sklearn library. It allows to test different hyperparameters and choose the best ones.

Without use of NLP, all models were off by 20k-30k dollars. It is a huge error.

# Evaluation

The data itself is not very good for predicting the salary. Although the dataset is quite big, its features are not very good for predicting. informations like job title, company name or job description are not very good for predicting the salary. Different features like domain, application status or if company page is listed were not very useful.

The idea itself looks very interesting and further research could be done. The dataset could be improved by adding more features. The use of NLP (Natural Language Processing) could be used for other features too.
