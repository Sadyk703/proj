# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 18:57:20 2021

@author: UmurzakovSI
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

os.chdir(r'C:\Users\UmurzakovSI\Desktop\Final\Active\new\PD Modelling\122021\0612')

exec(open(r'C:\Users\UmurzakovSI\Desktop\Final\Modules_and_packages.py').read())
exec(open(r'C:\Users\UmurzakovSI\Desktop\Final\All_functions.py').read())

# =============================================================================
# 
# =============================================================================
df_final_join = pd.read_pickle('df_final_join_new.pkl')
# corrs(df_final_join_new)

df_final_join = df_final_join.drop(['sales_to_cash', 'sales_to_cash_PC_PC', 'current_as_to_balance_currency', 'current_as_to_balance_currency_PC'], axis=1)

df_final_join_wo = cap_data(df_final_join)

df_final_join = df_final_join_wo.copy()


def namers(x):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if x in ('1'):
        return(0)
    elif x in ('0'):
        return(1)

df_final_join['Target_var'] = df_final_join['Target'].apply(lambda x: namers(str(x)))
df_final_join = df_final_join.drop('Target', axis=1)



df_final_join  = df_final_join.rename(columns={
                                 'X40':'short_term_debt_to_total_debt',
                                 'X31':'WC_to_total_assets',
                                 'X11': 'cash_&_financial_investments_to_current_assets',
                                 'X32': 'sales_to_cash',
                                 'X35': 'WC_to_equity',
                                 'X3': 'acc_receivable_to_assets',
                                 'X44': 'current_as_to_balance_currency',
                                 'X18': 'gross_profitability',
                                 'X16': 'NI_to_long_term_debt_&_short_term_debt',
                                 'X54':'EBIT_to_percents_to_be_paid',
                                 'X19':'current_assets_to_liab',
                                 'X4': 'EBITM',   
                                 'X52': 'debt_to_capital',
                                 'X38':'equity_to_borrowed capital',
                                 'X40_change': 'short_term_debt_to_total_debt_PC',
                                 'X31_change': 'WC_to_total_assets_PC',
                                 'X11_change':'cash_&_financial_investments_to_current_asset_PC',
                                 'X32_change': 'sales_to_cash_PC_PC',
                                 'X35_change': 'WC_to_equity_PC',
                                 'X3_change': 'acc_receivable_to_assets_PC',
                                 'X44_change': 'current_as_to_balance_currency_PC',
                                 'X18_change': 'gross_profitability_PC',
                                 'X16_change': 'NI_to_long_term_debt_&_short_term_debt_PC',
                                 'X54_change':'EBIT_to_percents_to_be_paid_PC',
                                 'X19_change':'current_assets_to_liab_PC',
                                 'X4_change': 'EBITM_PC',   
                                 'X52_change': 'debt_to_capital_PC',
                                 'X38_change':'equity_to_borrowed capital_PC',                                 
                                 'd_1age':'d_1yearage_of_company',
                                 'd_3age':'d_3yearage_of_company'})


def build_matrix(rows, cols):
    '''
    

    Parameters
    ----------
    rows : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.

    Returns
    -------
    matrix : TYPE
        DESCRIPTION.

    '''
    matrix = []

    for r in range(0, rows):
        matrix.append([int(0)  for c in range(0, cols)])
        

    return matrix

mat =  pd.DataFrame(build_matrix(2**25, 25))

mat.index = mat.index + 1
mat.columns = mat.columns + 1
temp = 0

for col in tqdm(mat.columns):
    for row in mat.index:
        if round(int(row)/2**(col - 1)) == int(row)/2**(col - 1):
            temp = 1 - temp
        mat.loc[row, col] = temp
        
mat['sum'] = mat.sum(axis=1)       

mat_10 = mat[mat['sum']==10] 

mat.to_pickle('mat_25_25.pkl')
mat_10.to_pickle('mat_10_25.pkl')
mat_frac_10_25 = mat_10.sample(frac=0.01)
mat_frac_10_25.to_pickle('mat_frac_10_25_1%.pkl')
mat_10 = pd.read_pickle('mat_frac_10_25_1%.pkl')

mat_10 = mat_10.drop(columns=['sum'])



# Essence analysis of features
var_essence = pd.DataFrame(index = df_final_join.columns)
var_essence['essence'] =  np.random.choice([-1, 0, 1], var_essence.shape[0])
var_essence.to_excel('var_essence_25.xlsx')
var_essence = pd.read_excel('var_essence_25.xlsx')


# Just check gini for whole dataset
X = df_final_join.drop(['Target_var'], axis = 1)
y = df_final_join['Target_var']

X_train_sp, X_test_sp, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42, stratify = y)

X_train_sp, X_test_sp = X_train_sp.copy(), X_test_sp.copy()

# Standartization of DataFrame 
X_train_sp_sl = X_train_sp.drop(['d_1yearage_of_company', 'd_3yearage_of_company'], axis=1)
X_test_sp_sl = X_test_sp.drop(['d_1yearage_of_company', 'd_3yearage_of_company'], axis=1)

df_final_join_sl = pd.DataFrame()
df_final_join_sl['Код налогоплательщика'] = X_train_sp_sl.index

X_train_scaled = stand_scaler(X_train_sp_sl)
X_train_scaled['Код налогоплательщика'] = df_final_join_sl['Код налогоплательщика']

df_dummy= pd.DataFrame(index = X_train_sp.index)
df_dummy['d_1yearage_of_company'] = X_train_sp['d_1yearage_of_company']
df_dummy['d_3yearage_of_company'] = X_train_sp['d_3yearage_of_company']
df_dummy.reset_index(drop=False, inplace=True)

X_train_scaled = pd.merge(X_train_scaled, df_dummy, on=['Код налогоплательщика'], how='inner')

df_final_join_sl_ts = pd.DataFrame()
df_final_join_sl_ts['Код налогоплательщика'] = X_test_sp_sl.index
X_test_scaled = stand_scaler(X_test_sp_sl) 
X_test_scaled['Код налогоплательщика'] = df_final_join_sl_ts['Код налогоплательщика']


df_dummy= pd.DataFrame(index = X_test_sp.index)
df_dummy['d_1yearage_of_company'] = X_test_sp['d_1yearage_of_company']
df_dummy['d_3yearage_of_company'] = X_test_sp['d_3yearage_of_company']
df_dummy.reset_index(drop=False, inplace=True)

X_test_scaled = pd.merge(X_test_scaled, df_dummy, on=['Код налогоплательщика'], how='inner')


X_train_scaled  = X_train_scaled.rename(columns={
                                 0:'short_term_debt_to_total_debt',
                                 1:'WC_to_total_assets',
                                 2: 'cash_&_financial_investments_to_current_assets',
                                 # 3: 'sales_to_cash',
                                 3: 'WC_to_equity',
                                 4: 'acc_receivable_to_assets',
                                 # 5: 'current_as_to_balance_currency',
                                 5: 'gross_profitability',
                                 6: 'NI_to_long_term_debt_&_short_term_debt',
                                 7:'EBIT_to_percents_to_be_paid',
                                 8:'current_assets_to_liab',
                                 9: 'EBITM',   
                                 10: 'debt_to_capital',
                                 11:'equity_to_borrowed capital',
                                 12: 'short_term_debt_to_total_debt_PC',
                                 13: 'WC_to_total_assets_PC',
                                 14:'cash_&_financial_investments_to_current_asset_PC',
                                 # 16: 'sales_to_cash_PC_PC',
                                 15: 'WC_to_equity_PC',
                                 16: 'acc_receivable_to_assets_PC',
                                 # 18: 'current_as_to_balance_currency_PC',
                                 17: 'gross_profitability_PC',
                                 18: 'NI_to_long_term_debt_&_short_term_debt_PC',
                                 19:'EBIT_to_percents_to_be_paid_PC',
                                 20:'current_assets_to_liab_PC',
                                 21: 'EBITM_PC',   
                                 22: 'debt_to_capital_PC',
                                 23:'equity_to_borrowed capital_PC',                                 
                                 24:'d_1yearage_of_company',
                                 25:'d_3yearage_of_company'})


X_train_scaled.index =X_train_scaled['Код налогоплательщика']
X_train_scaled = X_train_scaled.drop('Код налогоплательщика', axis=1)


X_test_scaled  = X_test_scaled.rename(columns={
                                 0:'short_term_debt_to_total_debt',
                                 1:'WC_to_total_assets',
                                 2: 'cash_&_financial_investments_to_current_assets',
                                 # 3: 'sales_to_cash',
                                 3: 'WC_to_equity',
                                 4: 'acc_receivable_to_assets',
                                 # 5: 'current_as_to_balance_currency',
                                 5: 'gross_profitability',
                                 6: 'NI_to_long_term_debt_&_short_term_debt',
                                 7:'EBIT_to_percents_to_be_paid',
                                 8:'current_assets_to_liab',
                                 9: 'EBITM',   
                                 10: 'debt_to_capital',
                                 11:'equity_to_borrowed capital',
                                 12: 'short_term_debt_to_total_debt_PC',
                                 13: 'WC_to_total_assets_PC',
                                 14:'cash_&_financial_investments_to_current_asset_PC',
                                 # 16: 'sales_to_cash_PC_PC',
                                 15: 'WC_to_equity_PC',
                                 16: 'acc_receivable_to_assets_PC',
                                 # 18: 'current_as_to_balance_currency_PC',
                                 17: 'gross_profitability_PC',
                                 18: 'NI_to_long_term_debt_&_short_term_debt_PC',
                                 19:'EBIT_to_percents_to_be_paid_PC',
                                 20:'current_assets_to_liab_PC',
                                 21: 'EBITM_PC',   
                                 22: 'debt_to_capital_PC',
                                 23:'equity_to_borrowed capital_PC',                                 
                                 24:'d_1yearage_of_company',
                                 25:'d_3yearage_of_company'})

X_test_scaled.index =X_test_scaled['Код налогоплательщика']
X_test_scaled = X_test_scaled.drop('Код налогоплательщика', axis=1)


# combination of 20 000 models with 26 var in Rstudio (d1 <- t(combn(26, 9)), head(d), final1 <- d1[1:20000, ])
mat_10_com = pd.read_excel('data9.xlsx')



df_final_join1 = df_final_join.drop('Target_var', axis=1)
cols_var = df_final_join1.columns.tolist()


# =============================================================================
# Scaled
# Create new DataFrame for result generation
Meta_df_sc = pd.DataFrame(index=range(5000), columns=['result', 'reason'])
# create dictionary
keys = range(5000)
dict_models_sk_sc = dict.fromkeys(keys)
dict_models_sk_coef_sc = dict.fromkeys(keys)  
# Dict with statmodels
dict_models_stat_params_sc = dict.fromkeys(keys) 
dict_models_stat_pvalues_sc = dict.fromkeys(keys)
dict_models_stat_summary_sc = dict.fromkeys(keys)

# Create df for the results of the probs
df_probs_sc = pd.DataFrame(index = X_test_scaled.index)
# df_probs_sc.hist()


for i in tqdm(range(0, 5000)):
    try:
        cols = [(cols_var[mat_10_com.loc[i, col]]) for col in mat_10_com.columns]

                
        X_train = X_train_scaled[cols]
        X_test = X_test_scaled[cols]    
        dict_models_sk_sc[i] = logit_loop(X_train, y_train, X_test, y_test, i)[1]
        dict_models_sk_coef_sc[i] = logit_loop(X_train, y_train, X_test, y_test, i)[5]
        df_probs_sc['prob_{}'.format(i)] =  logit_loop(X_train, y_train, X_test, y_test, i)[4]
        Meta_df_sc.loc[i, 'gini'] = round(logit_loop(X_train, y_train, X_test, y_test, i)[0], 3)
        dict_models_stat_params_sc[i] = logit_loop(sm.add_constant(X_train), y_train, sm.add_constant(X_test), y_test, i)[2]
        dict_models_stat_pvalues_sc[i] = logit_loop(sm.add_constant(X_train), y_train, sm.add_constant(X_test), y_test, i)[3]
        dict_models_stat_summary_sc[i] = logit_loop(X_train, y_train, X_test, y_test, i)[6]

        list_sign = []
        list_pvalues = []
        for j in range(len(X_train.columns)):
            k=0
            while X_train.columns[j]!= var_essence.loc[k, 'Features']:
                k+=1
            
            if dict_models_stat_params_sc[i][j] * var_essence.loc[k, 'essence']<0:
                
                list_sign.append(True)
            else:
                list_sign.append(False) 
            
            if logit_loop(X_train, y_train, X_test, y_test, i)[3][j]>0.1:
                list_pvalues.append(True)
            else:
                list_pvalues.append(False)
   
        if Meta_df_sc.loc[i, 'gini'] < 0.45:
            Meta_df_sc.loc[i, 'result'] = 0
            Meta_df_sc.loc[i, 'reason'] = '1.gini'
            continue
        elif any(list_sign):
            Meta_df_sc.loc[i, 'result'] = 0
            Meta_df_sc.loc[i, 'reason'] = '2.coef_sign'
            
            continue
        elif any(list_pvalues):
            Meta_df_sc.loc[i, 'result'] = 0
            Meta_df_sc.loc[i, 'reason'] = '3.pvalue'
                 
            continue
        else:
            Meta_df_sc.loc[i, 'result'] = 1
            Meta_df_sc.loc[i, 'reason'] = 'ok'  
    except:
        Meta_df_sc.loc[i, 'result'] = 0
        Meta_df_sc.loc[i, 'reason'] = 'Error'
        
yess_sc = Meta_df_sc[Meta_df_sc['result']==1]
yess_sc.to_pickle('yess_sc.pkl')   
# =============================================================================


corrs(X_train_scaled)


models = [1309, 2083, 3057, 3268, 3317, 3590, 3663]

final_dict_sk = {x:  dict_models_sk_sc[x] for x in models}
final_dict_models_stat_params = {x:  dict_models_stat_params_sc[x] for x in models}
final_dict_models_stat_pvalues = {x:  dict_models_stat_pvalues_sc[x] for x in models}
final_dict_models_stat_summary = {x:  dict_models_stat_summary_sc[x] for x in models}


df_2083 = pd.DataFrame(index = lr_2_train.index)

df_2083['2083'] = logit_final(lr_2_train, y_train)[4]



# =============================================================================
# 0312_Manipulation of Data
# =============================================================================

def ranks(x):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
  
    if 0 <= x <= 0.0019:
        return('Baa2')
    elif 0.0019 <= x <= 0.0026:
        return('Baa3')
    elif 0.0026 <= x <= 0.0046:
        return('Ba1')
    elif 0.0046 <= x <= 0.0072:
        return('Ba2')
    elif 0.0072 <= x <=  0.0101:
        return('Ba3')
    elif 0.0101 <= x <= 0.0193:
        return('B1')
    elif 0.0193 <= x <=  0.0308:
        return('B2')
    elif 0.0308 <= x <= 0.0394:
        return('B3')
    elif 0.0394 <= x <=  0.0625:
        return('Caa1')
    elif 0.0625 <= x <= 0.1383:
        return('Caa2')
    elif  0.1383 <= x <=  0.2823:
        return('Caa3') 
    elif 0.2823 <= x <= 1:
        return('Ca-C') 


df_final_score = pd.DataFrame()
df_final_score['2083'] = pd.concat([df_2083['2083'], df_probs_finals['2083']])
df_final_score['Moody_s DR'] = df_final_score['2083'].apply(lambda x: ranks(x))
df_final_score['Target_var'] = df_final_join['Target_var']
# Calculate the difference of the actual and forecasted PD score
df_final_score['Actual-Forecasted_PD'] = df_final_join['Target_var'] - df_final_score['2083']
df_final_score.to_excel('df_final_score.xlsx')

df_Caa1 = df_final_score.get_group('Caa1')

df_final_join['Target_var'].to_excel('target_var.xlsx')
df_final_score['2083'].to_excel('2083_model.xlsx') 


df_groupby_ranks =  pd.DataFrame()
df_groupby_ranks = df_final_score.groupby(by=["Moody_s DR"]).Target_var.value_counts()
df_PD_rate = pd.DataFrame(df_groupby_ranks)
df_PD_rate.to_excel('PD_ranks.xlsx')
df_PD_rate_ex = pd.read_excel('PD_ranks.xlsx')


# df_ser =pd.DataFrame(df_groupby_ranks) 
# df_ser.reset_index(drop=False, inplace=True)
# df_ser.to_excel('Ranks_all.xlsx')
# dfss = df_final_score.groupby(by="Moody_s DR")
# df_Caa1 = dfss.get_group('Caa1')
# df_ranks =  pd.DataFrame(df_Caa1)


# =============================================================================
# For Train dataset
# create dictionary
keys = range(10)
dict_models_sk = dict.fromkeys(keys)
dict_models_sk_coef = dict.fromkeys(keys)  
# Dict with statmodels
dict_models_stat_params = dict.fromkeys(keys) 
dict_models_stat_pvalues = dict.fromkeys(keys)
dict_models_stat_summary = dict.fromkeys(keys)

# Create df for the results of the probs
df_probs_tr = pd.DataFrame(index = X_train_scaled.index)
df_probs_tr.hist()

lists = mat_10_com.query('index == [1309, 2083, 3057, 3268, 3317, 3590, 3663]')
for i in tqdm(lists.index):
        cols = [(cols_var[mat_10_com.loc[i, col]]) for col in mat_10_com.columns]
                            
        X_train = X_train_scaled[cols]

                  
        dict_models_sk[i] = logit_loop_train(X_train, y_train, i)[1]
        dict_models_sk_coef[i] = logit_loop_train(X_train, y_train, i)[5]
        df_probs_tr['prob_{}'.format(i)] =  logit_loop_train(X_train, y_train, i)[4]
        yess_sc.loc[i, 'gini_train'] = round(logit_loop_train(X_train, y_train, i)[0], 3)
        dict_models_stat_params[i] = logit_loop_train(sm.add_constant(X_train), y_train, i)[2]
        dict_models_stat_pvalues[i] = logit_loop_train(sm.add_constant(X_train), y_train, i)[3]
        dict_models_stat_summary[i] = logit_loop_train(X_train, y_train, i)[6]
       
# =============================================================================


# =============================================================================
# For Train dataset
# Create df for the results of the probs
df_probs_alldf = pd.DataFrame(index = X_train_scaled.index)

lists = mat_10_com.query('index == [1309, 2083, 3057, 3268, 3317, 3590, 3663]')
for i in tqdm(lists.index):
        cols = [(cols_var[mat_10_com.loc[i, col]]) for col in mat_10_com.columns]
                            
        X_alldf = X[cols]
                  
        # dict_models_sk[i] = logit_loop_train(X_alldf, y, i)[1]
        # dict_models_sk_coef[i] = logit_loop_train(X_alldf, y, i)[5]
        df_probs_alldf['prob_{}'.format(i)] =  logit_loop_train(X_alldf, y, i)[4]
        yess_sc.loc[i, 'gini_alldf'] = round(logit_loop_train(X_alldf, y, i)[0], 3)
        # dict_models_stat_params[i] = logit_loop_train(sm.add_constant(X_alldf), y, i)[2]
        # dict_models_stat_pvalues[i] = logit_loop_train(sm.add_constant(X_alldf), y, i)[3]
        # dict_models_stat_summary[i] = logit_loop_train(X_train, y_train, i)[6]
       
# =============================================================================
df_probs_alldf.hist()

# =============================================================================
# Accuracy analysis
#**Accuracy**: $(TP + TN)/(TP + TN + FP + FN)$ -- the fraction of predictions our model got right

#**Precision**: $TP/ (TP + FP)$ -- What proportion of positive identifications was actually correct?

#**Recall**: $TP / (TP + FN)$ -- What proportion of actual positives was identified correctly?

model_acc = dict_models_sk[2083]

from sklearn.metrics import confusion_matrix, classification_report

dict_models_sk[2083] = logit_loop_train(lr_2_train, y_train, i)[1]

print('\nLogistic regression')
print(confusion_matrix(
    y_train, 
    dict_models_sk[2083].predict(lr_2_train)
))

print('\nLogistic regression')
print(confusion_matrix(y_train, dict_models_sk[2083].predict(lr_2_train), normalize='all'))


print('\nLogistic regression')
print(classification_report(y_train, dict_models_sk[2083].predict(lr_2_train)))

X_all = pd.concat([lr_2_train, lr_2_test])

print('\nLogistic regression')
print(classification_report(y, dict_models_sk[2083].predict(X_all)))

# =============================================================================


# =============================================================================
# 0612: Model  calibration
# =============================================================================


# Model
import optbinning.OptimalBinning as opt
import math
df_final_score['2083_arc'] = float(0)
# Переобразование
for i in tqdm(df_final_score.index):
    df_final_score['2083_arc'][i] = float(math.log((df_final_score['2083'][i])/(1-df_final_score['2083'][i])))


df_final_score['david'] = df_final_score['2083'].apply(lambda x: math.log(x/(1-x)))




df_final_score['2083_arc_bin'] =pd.cut(df_final_score['2083_arc'], 20)


from sklearn.calibration import calibration_curve
lr_y, lr_x = calibration_curve(y, df_final_score['2083'], n_bins=15)
fig, ax = plt.subplots()
plt.plot(lr_x, lr_y, marker='o', linewidth=1, label = 'lr')
line = matplotlib.lines.Line2D([0,1], [0,1], color = 'black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Calibration plot for Retail data')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability in each bin')
plt.legend();
plt.show()

sddd = plot_roc_curve(y, df_final_score['2083'])



df_final_score['2083']


plt.plot(df_final_score['david'], df_final_score['2083'])

logit_final(df_final_score['david'], df_final_score['Target_var'])

model_1 = sm.Logit(endog=df_final_score['Target_var'].astype(float), exog=sm.add_constant(df_final_score['david']).astype(float)).fit(disp=False, method='newton')
model_1.summary()





os.chdir(r'C:\Users\UmurzakovSI\Desktop\Final\Active\new\PD Modelling\122021\0712')
df_final_score_0712 = pd.read_excel('df_final_score.xlsx')
df_final_score_0712.index = df_final_score_0712['Код налогоплательщика']
df_final_score_0712 = df_final_score_0712.drop('Код налогоплательщика', axis=1)
df_final_score_0712['Score'].hist()
df_final_score_0712['2083'].hist()
df_final_score_0712['2083_arc'].hist()


def ranks_bin(x):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
  
    if 0 <= x <= 0.0019:
        return(0.0016)
    elif 0.0019 <= x <= 0.0026:
        return()
    elif 0.0026 <= x <= 0.0046:
        return('Ba1')
    elif 0.0046 <= x <= 0.0072:
        return('Ba2')
    elif 0.0072 <= x <=  0.0101:
        return('Ba3')
    elif 0.0101 <= x <= 0.0193:
        return('B1')
    elif 0.0193 <= x <=  0.0308:
        return('B2')
    elif 0.0308 <= x <= 0.0394:
        return('B3')
    elif 0.0394 <= x <=  0.0625:
        return('Caa1')
    elif 0.0625 <= x <= 0.1383:
        return('Caa2')
    elif  0.1383 <= x <=  0.2823:
        return('Caa3') 
    elif 0.2823 <= x <= 1:
        return('Ca-C') 

 0.0016 
 0.0023 
 0.0029 
 0.0062 
 0.0083 
 0.0118 
 0.0268 
 0.0348 
 0.0440 
 0.0810 
 0.1957 
 0.3690

for bins(df):
    df = 



from skorecard import datasets
from skorecard.bucketers import OptimalBucketer

X, y = datasets.load_uci_credit_card(return_X_y=True)



X_buck = df_final_score_0712['Score'].to_frame()
y_buck = df_final_score_0712['Target_var']
bucketer = OptimalBucketer(variables = ['Score'])
bucks = bucketer.fit_transform(X_buck, y_buck)



df_final_score_0712['Score_bin'] =pd.cut(df_final_score_0712['Score'], 12)

# =============================================================================
# Scorecard development - 0712
# =============================================================================


# Store the column names in X_train as a list
feature_name = lr_2_train.columns.values
# Create a summary table of our logistic regression model
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
# Create a new column in the dataframe, called 'Coefficients', with row values the transposed coefficients from the 'LogisticRegression' model
summary_table['Coefficients'] = np.transpose(dict_models_sk_coef[2083])
# Increase the index of every row of the dataframe with 1 to store our model intercept in 1st row
summary_table.index = summary_table.index + 1
# Assign our model intercept to this new row
summary_table.loc[0] = ['Intercept',  dict_models_sk[2083].intercept_[0]]
# Sort the dataframe by index
summary_table.sort_index(inplace = True)
summary_table

def ranks(x):
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
  
    if 0 <= x <= 0.0019:
        return('Baa2')
    elif 0.0019 <= x <= 0.0026:
        return('Baa3')
    elif 0.0026 <= x <= 0.0046:
        return('Ba1')
    elif 0.0046 <= x <= 0.0072:
        return('Ba2')
    elif 0.0072 <= x <=  0.0101:
        return('Ba3')
    elif 0.0101 <= x <= 0.0193:
        return('B1')
    elif 0.0193 <= x <=  0.0308:
        return('B2')
    elif 0.0308 <= x <= 0.0394:
        return('B3')
    elif 0.0394 <= x <=  0.0625:
        return('Caa1')
    elif 0.0625 <= x <= 0.1383:
        return('Caa2')
    elif  0.1383 <= x <=  0.2823:
        return('Caa3') 
    elif 0.2823 <= x <= 1:
        return('Ca-C') 


df_final_score = pd.DataFrame()
df_final_score['2083'] = pd.concat([df_2083['2083'], df_probs_finals['2083']])
df_final_score['Moody_s DR'] = df_final_score['2083'].apply(lambda x: ranks(x))
df_final_score['Target_var'] = df_final_join['Target_var']
# Calculate the difference of the actual and forecasted PD score
df_final_score['Actual-Forecasted_PD'] = df_final_join['Target_var'] - df_final_score['2083']
df_final_score.to_excel('df_final_score.xlsx')

df_Caa1 = df_final_score.get_group('Caa1')

df_final_join['Target_var'].to_excel('target_var.xlsx')
df_final_score['2083'].to_excel('2083_model.xlsx') 


df_groupby_ranks_0712 =  pd.DataFrame()
df_groupby_ranks_0712 = df_final_score_0712.groupby(by=["Moody_s DR"]).Score.value_counts()
df_PD_rate_0712 = pd.DataFrame(df_groupby_ranks_0712)
df_PD_rate_0712.to_excel('PD_ranks_0712.xlsx')
df_PD_rate_0712.reset_index(drop=False, inplace=True)

df_PD_rate_0712.to_excel('PD_ranks_0712.xlsx')
df_PD_rate_ex_0712 = pd.read_excel('PD_ranks_0712.xlsx')
df_PD_rate_ex_0712 = df_PD_rate_ex_0712.fillna(method="ffill")


df_PD_rate_ex_0712_new = df_PD_rate_ex_0712.groupby(by=["Moody_s DR"]).Score.min_max()
df_PD_rate_ex_0712_new.to_excel('df_PD_rate_ex_0712_new.xlsx')
df_PD_rate_ex_0712_new_max = df_PD_rate_ex_0712.groupby(by=["Moody_s DR"]).Score.max()
df_PD_rate_ex_0712_new_max.to_excel('df_PD_rate_ex_0712_new_max.xlsx')


# df_ser =pd.DataFrame(df_groupby_ranks) 
# df_ser.reset_index(drop=False, inplace=True)
# df_ser.to_excel('Ranks_all.xlsx')
# dfss = df_final_score.groupby(by="Moody_s DR")
# df_Caa1 = dfss.get_group('Caa1')
# df_ranks =  pd.DataFrame(df_Caa1)










