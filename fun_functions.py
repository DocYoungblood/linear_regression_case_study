from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import PredictionError, ResidualsPlot
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.datasets import load_diabetes
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from yellowbrick.features import JointPlotVisualizer
sns.set_theme()


'''
IMPORT INSTRUCTIONS

--import fun_functions as fun--

call as fun.(*fun*ction name here)

Have fun!

'''

def lasso_model(X, y):# Lasso with 5 fold cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40)
    #model = LassoCV(cv=10, random_state=40, max_iter=10000)
    model = make_pipeline(StandardScaler(), LassoCV(cv=10, random_state=40, max_iter=10000))
    # Fit model
    model.fit(X_train, y_train)

    model[1].alpha_

    lasso_best = Lasso(alpha=model[1].alpha_)
    lasso_best.fit(X_train, y_train)
    print('Best coeficient - X value:', list(zip(lasso_best.coef_, X)))
    print('R squared training set', round(lasso_best.score(X_train, y_train)*100, 2))
    print('R squared test set', round(lasso_best.score(X_test, y_test)*100, 2))
    print('MSE:', mean_squared_error(y_test, lasso_best.predict(X_test)))
    return model, lasso_best

def ridge_model(X, y):# Lasso with 5 fold cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40)
    model1 = RidgeCV(cv=10)

    # Fit model
    model1.fit(X_train, y_train)

    model1.alpha_

    ridge_best = Ridge(alpha=model1.alpha_)
    ridge_best.fit(X_train, y_train)
    print('Best coeficient - X value:', list(zip(ridge_best.coef_, X)))

    print('R squared training set', round(ridge_best.score(X_train, y_train)*100, 2))
    print('R squared test set', round(ridge_best.score(X_test, y_test)*100, 2))
    print(mean_squared_error(y_test, ridge_best.predict(X_test)))
    return model1, ridge_best

def mse_on_fold_plot(model, ymin, ymax):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.semilogx(model.alphas_, model.mse_path_, ":")
    
    ax1.plot(
        model.alphas_ ,

        model.mse_path_.mean(axis=-1),
        "k",
        label="Average across the folds",
        linewidth=2,
    )
    ax1.axvline(
        model.alpha_, linestyle="--", color="k", label=f"Optimal alpha: CV estimate ({model.alpha_:.4f})"
    )

    ax1.legend()
    ax1.set_xlabel("alphas")
    ax1.set_ylabel("Mean square error")
    ax1.set_title("Mean square error on each fold:\nBanking Features")
    ax1.axis("tight")

    ymin, ymax = 0, 20
    ax1.set_ylim(ymin, ymax);


def coeficients_plot(alphas, X, y, model):
    coefs = []
    for a in alphas:
        model.set_params(alpha=a)
        model.fit(X, y)
        coefs.append(model.coef_)
    
    ax = plt.gca()
    
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('Standardized Coefficients')
    plt.title('Lasso coefficients as a function of alpha')

if __name__ == '__main__':
    pass