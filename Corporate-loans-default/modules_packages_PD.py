# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:37:07 2021

@author: UmurzakovSIl
"""

import pandas as pd
import ast
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

import matplotlib

import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.metrics import plot_roc_curve
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency

from sklearn.experimental import enable_iterative_imputer #MUST IMPORT THIS
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import math


from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import f_oneway
# Mann-Whitney U Test
from scipy.stats import mannwhitneyu
# Kruskal-Wallis H-test
from scipy.stats import kruskal
from matplotlib import pyplot
from scipy.stats import kstest
import statsmodels.api as sm

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
