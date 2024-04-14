# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:03:24 2021

@author: UmurzakovSI
"""

import pandas as pd
import numpy as np
import scipy as sc
import sklearn as sk
import os
from sqlalchemy.types import Integer
from tqdm import tqdm
import random 
from random import randrange
import re 
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
import statsmodels.formula.api as sm



# =============================================================================
# Corr calculation
# =============================================================================

def corrs(X_scaled):  
    """
    

    Parameters
    ----------
    X_scaled : df for  corr calculation

    Returns
    -------
    Show plt

    """
    plt.figure(figsize=(14,12))
    sns.heatmap(X_scaled.corr(),linewidths=0.1,vmax=1.0, 
                square=True,  linecolor='white', annot=True)
    return plt.show()
    
# =============================================================================
# Outliers removing
# =============================================================================

def remove_outlier(df_in, col_name):
    """
    

    Parameters
    ----------
    df_in : df for .
    col_name : the name of column that  you want to remove outliers.

    Returns
    -------
    df_out : return the value after removing.

    """
    q1 = df_in[col_name].quantile(0.01)
    q3 = df_in[col_name].quantile(0.99)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out




def outlier(df):
    
    df_final = df.copy()
    
    for col in df.columns:
        lower = df[col].quantile(0.001)
        upper = df[col].quantile(0.999)
        
        df_final = df_final.loc[((df_final[col]>lower ) & (df_final[col]<upper))]
    
    return df_final
    
    
# df = df_final_join



def cap_data(df):
    for col in df.columns:
        print("capping the ",col)
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles = df[col].quantile([0.01,0.99]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col]=df[col]
    return df




def outlier_detect(df):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    for i in df.columns:
        Q1=df[i].quantile(0.01)
        Q3=df[i].quantile(0.99)
        # IQR=Q3 - Q1
        # LTV=Q1 - 1.5 * IQR
        # UTV=Q3 + 1.5 * IQR
        delete
        x=np.array(df[i])
        p=[]
        for j in x:
            if j < LTV or j>UTV:
                p.append(df[i].median())
            else:
                p.append(j)
        df[i]=p
    return df




# =============================================================================
# WoE Transformation and Feature Engineering
# =============================================================================

def woeTransform(df):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    '''
    zeros = len(df[df['Target'] == "0"].index)
    ones = len(df[df['Target'] == "1"].index)
    
    res = np.log(ones/zeros)
    return res


def woeTransform(df, numGood=144614, numBad=65118):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    numGood : TYPE, optional
        DESCRIPTION. The default is 144614.
    numBad : TYPE, optional
        DESCRIPTION. The default is 65118.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    '''
    zeros = len(df[df['Target'] == "0"].index)
    ones = len(df[df['Target'] == "1"].index)
    
    res = np.log((ones/numGood)/(zeros/numBad))
    return res

def woeData(data, col, q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    q : TYPE, optional
        DESCRIPTION. The default is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''
    
    # q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9]
    
    quantile = []
    for quan in q:
        quantile.append(data[col].quantile(quan))
    
    delet = []
    
    for i  in range(1,(len(q)-1)):
        if quantile[i] == 0:
            delet.append(i)
            
    if len(delet) > 1:
        quantile = quantile[-delet]
    
    quantile = sorted(list(set(quantile)))
          
    d_na = data[pd.isnull(data[col])]
    
    output = []
    
    for i in range(0,len(quantile)):
        if i == 0:
            d = data[data[col]<=quantile[i]]
            output.append(woeTransform(d))
        elif i == len(quantile):
            d = data[data[col]>quantile[i]  ]
            output.append(woeTransform(d))            
        else:    
            d = data[(data[col]>quantile[i-1]) & (data[col]<=quantile[i]) ]
            output.append(woeTransform(d))
    
    output.append(woeTransform(d_na))
    
    quantile.append(np.nan)
    
    result = pd.DataFrame( output, quantile).reset_index()
    result = result.rename(columns = {'index': "q", 0: "output"})
    return result



def trans(x, result):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    result : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if pd.isnull(x):
        r = result[pd.isnull(result['q'])].reset_index()
        return  r.loc[0, 'output']
    else:
        k = 0
        while x > result.loc[k, 'q']:
            k+=1
        return result.loc[k, 'output']


def woeColumn(data):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
       
    
    for cols in data_colnames:
        print(cols)
        result = woeData(data, cols, q)
        
        result = result[pd.notnull(result['output'])]
        
        data['temp'] = data[cols].apply(lambda x: trans(x, result))
        
        # # for i in data.index:
        # #     if pd.isnull(data.loc[i, cols]):
        # #         r = result[pd.isnull(result['q'])].reset_index()
        # #         data.loc[i, 'temp'] = r.loc[0, 'output']
        # #     else:
        # #         k = 0
        # #         while data.loc[i, cols]>result.loc[k, 'q']:
        # #             k+=1
        # #         data.loc[i, 'temp'] = result.loc[k, 'output']
        
        data = data.rename(columns={'temp': 'woe_{}'.format(cols)})
    
    return data


#First attempt to group for monotonicity
def add_woe(df, col, q =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9] ):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    q : TYPE, optional
        DESCRIPTION. The default is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9].

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    df = df.drop(columns=["woe_{}".format(col)])
    df[col] = X_train[col]
    df['temp']= 0
    
    res = woeData(df, col, q)
    df['temp'] = df[col].apply(lambda x:  trans(x, res))
    df = df.rename(columns = {'temp': "woe_{}".format(col)})
    df = df.drop(columns = [col])
    
    return df

# X_train2 = add_woe(X_train_woe, 'X1')

# X_train2 = add_woe(X_train2, 'X6', q =[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])


# =============================================================================
# WOE data manipulation
# =============================================================================


def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    '''
    

    Parameters
    ----------
    df_WoE : TYPE
        DESCRIPTION.
    rotation_of_x_axis_labels : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    '''
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels) 


def woe_ordered_continuous(df, continuous_variabe_name, y_df):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    continuous_variabe_name : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    df = pd.concat([df[continuous_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    #df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    #df['IV'] = df['IV'].sum()
    return df

# X_train_woe2  = round(X_train2, 1)
# X = X_train_woe.drop('Target',  axis=1)
# y =  X_train['Target'].to_frame().astype(int)

# df_x1 = woe_ordered_continuous(X, 'woe_X1', y)
# plot_by_woe(df_x1)

# =============================================================================
# Logit function
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def logits(x_df,  var, y_df):
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    model = LogisticRegression()
    model.fit(X=x_df, y=y_df)
    x_df['prob'] = model.predict_proba(x_df)[:,1]
    score = roc_auc_score(y_df, x_df['prob'])
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1
    #plt.plot(gini_1)
    dfd = x_df.groupby(by=[var]).mean()['prob']
    return dfd

# gini_1 = logits(X_train_woe, 'woe_X1', y)
# plt.plot(gini_1)

# gini_2 = logits(X_train_woe, 'woe_X1', y)
# plt.plot(gini_2)


# =============================================================================
# Standartization
# =============================================================================

def stand_scaler(X_train):
    '''
    

    Parameters
    ----------
    X_train0 : TYPE
        DESCRIPTION.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.

    '''
    ss = StandardScaler()
    X_train0 = pd.DataFrame(X_train.fillna(X_train.mean()))
    # X_train0 = X_train0.drop(columns=['X20'])
    X_train0 = X_train0[X_train0.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    # X_train1 = X_train0.drop('Target', axis=1)
    X_scaled =  pd.DataFrame(ss.fit_transform(X_train0))
    return X_scaled

# https://levelup.gitconnected.com/an-introduction-to-logistic-regression-in-python-with-statsmodels-and-scikit-learn-1a1fb5ce1c13
def logits(x_df, y_df, x_ts, y_ts):
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    import statsmodels.api as sm
    model = LogisticRegression()
    model.fit(X=x_df, y=y_df)
    model_1 = sm.Logit(endog=y_df.astype(float), exog=x_df.astype(float)).fit()
    x_ts['prob'] = model.predict_proba(x_ts)[:,1]
    score = roc_auc_score(y_ts, x_ts['prob'])
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1
    y_ts['prob'] = x_ts['prob']
    sd = print(model_1.summary())
    return GINI, sd, y_ts, model.coef_, model.intercept_



def logit_loop(x_df, y_df, x_ts, y_ts, k):
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    df_prob = pd.DataFrame(index = x_ts.index)
    import statsmodels.api as sm
    model = LogisticRegression(fit_intercept=True, solver='newton-cg', penalty = 'none')
    model_sk = model.fit(X=x_df.astype(float), y=y_df.to_frame().astype(float))
    model_1 = sm.Logit(endog=y_df.astype(float), exog=x_df.astype(float)).fit(disp=False, method='newton')
    df_prob['prob_{}'.format(k)] = model_sk.predict_proba(x_ts.astype(float))[:,1]
    score = roc_auc_score(y_ts.to_frame().astype(float), df_prob['prob_{}'.format(k)])
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1

    return GINI, model_sk,  model_1.params, model_1.pvalues, df_prob, model_sk.coef_, model_1.summary()



def logit_loop_train(x_df, y_df, k):
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    df_prob = pd.DataFrame(index = x_df.index)
    import statsmodels.api as sm
    model = LogisticRegression(fit_intercept=True, solver='newton-cg', penalty = 'none')
    model_sk = model.fit(X=x_df.astype(float), y=y_df.to_frame().astype(float))
    model_1 = sm.Logit(endog=y_df.astype(float), exog=x_df.astype(float)).fit(disp=False, method='newton')
    df_prob['prob_{}'.format(k)] = model_sk.predict_proba(x_df.astype(float))[:,1]
    score = roc_auc_score(y_df.to_frame().astype(float), df_prob['prob_{}'.format(k)])
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1

    return GINI, model_sk,  model_1.params, model_1.pvalues, df_prob, model_sk.coef_, model_1.summary()

def logit_RandSearch(x_df, y_df, k):
    
    
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(shuffle=True, n_splits=3)
    df_prob = pd.DataFrame(index = x_df.index)
    
    import statsmodels.api as sm
    from sklearn.model_selection import RandomizedSearchCV
    params = {
        'C': np.logspace(10, 50, 100),
        'penalty': 'none'
        }
    
    gs_lr = LogisticRegression(fit_intercept=True, solver='newton-cg', penalty = 'none')
    model = RandomizedSearchCV(
        estimator=gs_lr, 
        param_distributions=params, 
        cv=cv, 
        scoring='roc_auc',
        return_train_score=True)
    model_sk = model.fit(X=x_df.astype(float), y=y_df.to_frame().astype(float))
    model_1 = sm.Logit(endog=y_df.astype(float), exog=x_df.astype(float)).fit(disp=False, method='newton')
    df_prob['prob_{}'.format(k)] = model_sk.predict_proba(x_df.astype(float))[:,1]
    score = roc_auc_score(y_df.to_frame().astype(float), df_prob['prob_{}'.format(k)])
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1

    return GINI, model_sk,  model_1.params, model_1.pvalues, df_prob, model_sk.coef_, model_1.summary()




def logit_final(x_df, y_df):
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    import statsmodels.api as sm
    model = LogisticRegression(fit_intercept=True, solver='newton-cg', penalty = 'none')
    model_sk = model.fit(X=x_df.astype(float), y=y_df.to_frame().astype(float))
    model_1 = sm.Logit(endog=y_df.astype(float), exog=sm.add_constant(x_df).astype(float)).fit(disp=False, method='newton')
    df_prob = model_sk.predict_proba(x_df.astype(float))[:,1]
    score = roc_auc_score(y_df.to_frame().astype(float), df_prob)
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1
    return model_1.summary() #GINI, model_sk,  model_1.params, model_1.pvalues, df_prob








def log_stat(x_df, y_df):
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    model_2 = sm.Logit(endog=y_df.astype(float), exog=x_df.astype(float)).fit(disp=False)
    return model_2.summary()
    


def logits_1(x_df, y_df):
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    import statsmodels.api as sm
    model = LogisticRegression()
    model.fit(X=x_df, y=y_df)
    model_1 = sm.Logit(endog=y_df.astype(float), exog=x_df.astype(float)).fit()
    x_df['prob'] = model.predict_proba(x_df)[:,1]
    score = roc_auc_score(y_df, x_df['prob'])
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1
    y_df['prob'] = x_df['prob']
    sd = print(model_1.summary())
    return GINI, sd, y_df, model.coef_, model.intercept_



def logits_gini(x_df, y_df):
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    model = LogisticRegression()
    model.fit(X=x_df, y=y_df)
    x_df['prob'] = model.predict_proba(x_df)[:,1]
    score = roc_auc_score(y_df, x_df['prob'])
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1
    return GINI



def logit(x_df, y_df):
    
    
    '''
    

    Parameters
    ----------
    x_df : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.
    y_df : TYPE
        DESCRIPTION.

    Returns
    -------
    dfd : TYPE
        DESCRIPTION.

    '''
    import statsmodels.api as sm
    from sklearn.model_selection import RandomizedSearchCV
    params = {
        'C': np.logspace(-3, -0.2, 20),
        'penalty': ['l1', 'l2']
        }
    
    gs_lr = LogisticRegression(penalty='l2',C=0.01)
    model = RandomizedSearchCV(
        estimator=gs_lr, 
        param_distributions=params, 
        cv=cv, 
        scoring='roc_auc',
        return_train_score=True)
    model.fit(X=x_df, y=y_df)
    model_1 = sm.OLS(endog=y_df.astype(float), exog=x_df.astype(float)).fit()
    x_df['prob'] = model.predict_proba(x_df)[:,1]
    score = roc_auc_score(y_df, x_df['prob'])
    AUROC = np.mean(score)
    GINI = AUROC * 2 - 1
    y_df['prob'] = x_df['prob']
    sd = print(model_1.summary())
    return GINI, sd, y_df, model.best_estimator_, model.best_params_

# Stattistical tests for testing the difference
def t_test(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = stats.ttest_ind(y.astype(float), x.astype(float))[0]
    p_values = stats.ttest_ind(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec


# Paired Student’s t-Test
def t_paired_test(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = stats.ttest_rel(y.astype(float), x.astype(float))[0]
    p_values = stats.ttest_rel(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec

def f_anova(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = f_oneway(y.astype(float), x.astype(float))[0]
    p_values = f_oneway(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec


def f_test(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = f_classif(y.astype(float), x.astype(float))[0]
    p_values = f_classif(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec


def wilcox_test(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = wilcoxon(y.astype(float), x.astype(float))[0]
    p_values = wilcoxon(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec


# Mann-Whitney U Test
def Mann_Whitney(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = mannwhitneyu(y.astype(float), x.astype(float))[0]
    p_values = mannwhitneyu(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec




def kruskal_test(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = kruskal(y.astype(float), x.astype(float))[0]
    p_values = kruskal(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec




def friedman_test(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = friedmanchisquare(y.astype(float), x.astype(float))[0]
    p_values = friedmanchisquare(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec


def ks_test(y, x):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    t_tests : TYPE
        DESCRIPTION.
    p_values : TYPE
        DESCRIPTION.
    dec : TYPE
        DESCRIPTION.

    '''
    
    t_tests = kstest(y.astype(float), x.astype(float))[0]
    p_values = kstest(y.astype(float), x.astype(float))[1]
    if p_values < 0.05: 
        dec = 'good'
    else: 
        dec = 'bad'
    return t_tests, p_values, dec






def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)




# =============================================================================
# Feature selection procedure
# =============================================================================
# Forward selection procedute
import statsmodels.formula.api as sm
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

# Another method of forward  selection


# Backward  selection 
def backward_regression(X, y,
                           initial_list=[], 
                           threshold_in=0.01, 
                           threshold_out = 0.05, 
                           verbose=True):
    included=list(X.columns)
    while True:
        changed=False
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=False)
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included





# Loof for multiple regressions

def displaymetrics(code, dict_models, X_train, X_test, y_train, y_test):

    
            for model_instantiation in dict_models.iteritems():
                import pandas as pd
                import numpy as np
                from sklearn.metrics import roc_curve, auc
                from sklearn.metrics import accuracy_score, recall_score, precision_score
                from sklearn.model_selection import cross_val_score
                y_score = model_instantiation.fit(X_train, y_train).decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)

        # Traditional Scores

                y_pred = pd.DataFrame(model_instantiation.predict_proba(X_train)).reset_index(drop=True)
                Recall_Train,Precision_Train, Accuracy_Train  = recall_score(y_train, y_pred), precision_score(y_train, y_pred), accuracy_score(y_train, y_pred)
                y_pred = pd.DataFrame(model_instantiation.predict_proba(X_test)).reset_index(drop=True)
                Recall_Test = recall_score(y_test, y_pred)
                Precision_Test = precision_score(y_test, y_pred)
                Accuracy_Test = accuracy_score(y_test, y_pred)

        #Cross Validation
                cv_au = cross_val_score(model_instantiation, X_test, y_test, cv=30, scoring='roc_auc')
                cv_f1 = cross_val_score(model_instantiation, X_test, y_test, cv=30, scoring='f1')
                cv_pr = cross_val_score(model_instantiation, X_test, y_test, cv=30, scoring='precision')
                cv_re = cross_val_score(model_instantiation, X_test, y_test, cv=30, scoring='recall')
                cv_ac = cross_val_score(model_instantiation, X_test, y_test, cv=30, scoring='accuracy')
                cv_ba = cross_val_score(model_instantiation, X_test, y_test, cv=30, scoring='balanced_accuracy')
                cv_au_m, cv_au_std =  cv_au.mean() , cv_au.std() 
                cv_f1_m, cv_f1_std = cv_f1.mean() , cv_f1.std()
                cv_pr_m, cv_pr_std = cv_pr.mean() , cv_pr.std()
                cv_re_m, cv_re_std= cv_re.mean() , cv_re.std()
                cv_ac_m, cv_ac_std = cv_ac.mean() , cv_ac.std()
                cv_ba_m, cv_ba_std= cv_ba.mean() , cv_ba.std()
                cv_au, cv_f1, cv_pr =  (cv_au_m, cv_au_std),  (cv_f1_m, cv_f1_std), (cv_pr_m, cv_pr_std) 
                cv_re, cv_ac, cv_ba = (cv_re_m, cv_re_std), (cv_ac_m, cv_ac_std), (cv_ba_m, cv_ba_std)
                tuples = [cv_au, cv_f1, cv_pr, cv_re, cv_ac, cv_ba]
                tuplas = [0]*len(tuples)
                for i in range(len(tuples)):
                    tuplas[i] = [round(x,4) for x in tuples[i]]
                results = pd.DataFrame()
                results['Metrics'] = ['roc_auc', 'Accuracy_Train', 'Precision_Train', 'Recall_Train', 'Accuracy_Test', 
                              'Precision_Test','Recall_Test', 'cv_roc-auc (mean, std)', 'cv_f1score(mean, std)', 
                              'cv_precision (mean, std)', 'cv_recall (mean, std)', 'cv_accuracy (mean, std)', 
                              'cv_bal_accuracy (mean, std)']
                results.set_index(['Metrics'], inplace=True)
                results['Model_'+code[i]] = [roc_auc, Accuracy_Train, Precision_Train, Recall_Train, Accuracy_Test, 
                            Precision_Test, Recall_Test, tuplas[0], tuplas[1], tuplas[2], tuplas[3],
                           tuplas[4], tuplas[5]]

            return results   


import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np
from scipy.stats import chi2

# This could be made into a neat function of Hosmer-Lemeshow. 
def HosmerLemeshow (model,Y):
    pihat=model.predict()
    pihatcat=pd.cut(pihat, np.percentile(pihat,[0,25,50,75,100]),labels=False,include_lowest=True) #here we've chosen only 4 groups


    meanprobs =[0]*4 
    expevents =[0]*4
    obsevents =[0]*4 
    meanprobs2=[0]*4 
    expevents2=[0]*4
    obsevents2=[0]*4 

    for i in range(4):
       meanprobs[i]=np.mean(pihat[pihatcat==i])
       expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
       obsevents[i]=np.sum(Y[pihatcat==i])
       meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
       expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
       obsevents2[i]=np.sum(1-Y[pihatcat==i]) 


    data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
    data2={'expevents':expevents,'expevents2':expevents2}
    data3={'obsevents':obsevents,'obsevents2':obsevents2}
    m=pd.DataFrame(data1)
    e=pd.DataFrame(data2)
    o=pd.DataFrame(data3)
    
    # The statistic for the test, which follows, under the null hypothesis,
    # The chi-squared distribution with degrees of freedom equal to amount of groups - 2. Thus 4 - 2 = 2
    tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) 
    pvalue=1-chi2.cdf(tt,2)

    return pd.DataFrame([[chi2.cdf(tt,2).round(2), pvalue.round(2)]],columns = ["Chi2", "p - value"])
    



def hl_test(data, g):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data

    Input: dataframe(data), integer(num of subgroups divided)
    
    Output: float
    '''
    data_st = data.sort_values('prob')
    data_st['dcl'] = pd.qcut(data_st['prob'], g)
    
    ys = data_st['ViolentCrimesPerPop'].groupby(data_st.dcl).sum()
    yt = data_st['ViolentCrimesPerPop'].groupby(data_st.dcl).count()
    yn = yt - ys
    
    yps = data_st['prob'].groupby(data_st.dcl).sum()
    ypt = data_st['prob'].groupby(data_st.dcl).count()
    ypn = ypt - yps
    
    hltest = ( ((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn) ).sum()
    pval = 1 - chi2.cdf(hltest, g-2)
    
    df = g-2
    
    print('\n HL-chi2({}): {}, p-value: {}\n'.format(df, hltest, pval))


