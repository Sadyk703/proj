# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:15:54 2021

@author: UmurzakovSI
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2
import math
from datetime import datetime
import datetime as dt
from tqdm import tqdm


os.chdir(r'N:\_ЦУРКР_\2. Финансовые рынки\2. VaR\3. Расчеты\2610')

ytm = pd.read_excel('ytm_portf.xlsx')
ytm = ytm.set_index('TRADEDATE')
ytm = ytm.fillna(method = 'ffill')

duration = pd.read_excel('duration_portf.xlsx')
duration = duration.set_index('TRADEDATE')
duration = duration.interpolate(method='linear', axis=0)

close = pd.read_excel('close_portf.xlsx')
close = close.set_index('TRADEDATE')
close = close.fillna(method = 'ffill')

close_pc = close.pct_change()

# floater_days = pd.read_excel('floater mat days.xlsm', sheet_name='Sheet1')

bonds = pd.read_excel('securities.xlsx', sheet_name='Bonds')
bonds = bonds.set_index('ISIN')

bonds_portf = (bonds['Балансовая стоимость пакета(руб.)'] + bonds['Переоценка']).sum()
bonds_weights = (bonds['Балансовая стоимость пакета(руб.)'] + bonds['Переоценка'])/ bonds_portf



def return_calc(ind, data_price, data_volume):
    """
    calculates the return of portfolio depending on the data availabilty
    """
    
    df = data_price.loc[data_price.index == ind]
    if df.isnull().values.any() == True:
        nan_cols = [i for i in df.columns if df[i].isnull().any()]
        new_df = df.drop(nan_cols, axis = 1)
        df = new_df.copy()
        new_sspsd = data_volume.drop(nan_cols)
        # new_portf_volume = data_volume['Остаток на счете ВложЦБ руб.'].sum()
        new_portf_volume = (new_sspsd['Балансовая стоимость пакета(руб.)'] + new_sspsd['Переоценка']).sum()
        new_share = (new_sspsd['Балансовая стоимость пакета(руб.)'] + new_sspsd['Переоценка']) /new_portf_volume
        portf_return = 0
        for i in new_share.index:
            portf_return += new_df.loc[ind, i]*new_share[i]
        return (portf_return)
    else:
        new_portf_volume = (data_volume['Балансовая стоимость пакета(руб.)'] + data_volume['Переоценка']).sum()
        new_share = (data_volume['Балансовая стоимость пакета(руб.)'] + data_volume['Переоценка'])/new_portf_volume
        portf_return = 0
        for i in new_share.index:
            portf_return += df.loc[ind, i]*new_share[i]        
        return (portf_return)
    



close_pc['portfolioReturn'] = ''
for i in close_pc.index:
    close_pc.loc[i, 'portfolioReturn'] = return_calc(i, close_pc, bonds)

plt.plot(close_pc['portfolioReturn'])





# =============================================================
# Duration of floater rate bond 
# =============================================================
 
duration['date'] = duration.index

# for i in duration[duration.index > '2020-11-01 00:00:00'].index:
#     if duration.loc[i, 'date'] < floater_days.loc[0, 'Dates']:
#         continue
#     else:
#         k = 0
#         while duration.loc[i, 'date'] >= floater_days.loc[k, 'Dates']:
#             k += 1
#         duration.loc[i, 'RU000A102BV4'] = (floater_days.loc[k, 'Dates'] - duration.loc[i, 'date']).days



# =============================================================
# Modified duration calculcation
# =============================================================

modDur = pd.DataFrame(index = duration.index, columns = duration.columns)

# modified duration calculation
for col in modDur.columns:
    try:
        modDur[col] = duration[col]/(1+ ytm[col]/100)/365   
    except:
        None

modDur = modDur.drop(columns = 'date')
modDur = modDur.interpolate(method='linear', axis=1) #drop('date', axis =1).

# =============================================================
# Modified duration VaR calculation
# =============================================================    

def ModDurVaR(ind, Data_modDur, Data_ytm, Confidence_Interval = 0.99):
    """ Calculates VaR with modified duration
    """
    
    df = Data_ytm.loc[Data_ytm.index == ind]
    if df.isnull().values.any() == True:
        nan_cols = [i for i in df.columns if df[i].isnull().any()]
        new_bonds = bonds.drop(nan_cols)
        weights = (new_bonds['Балансовая стоимость пакета(руб.)'] + new_bonds['Переоценка'])/(new_bonds['Балансовая стоимость пакета(руб.)'] + new_bonds['Переоценка']).sum()
        Data_modDur = Data_modDur.drop(nan_cols, axis =1)
        Data_ytm = Data_ytm.drop(nan_cols, axis = 1)
        Data_ytm = Data_ytm[1:253]
        nan_cols_add = []
        for i in Data_ytm.columns:
            a = pd.isnull(Data_ytm[i])
            false_count = (~a).sum()
            if false_count <= 7:
                nan_cols_add.append(i)
        if len(nan_cols_add) > 0:
            Data_ytm = Data_ytm.drop(nan_cols_add, axis = 1)
            Data_modDur = Data_modDur.drop(nan_cols_add, axis= 1)
            new_bonds = new_bonds.drop(nan_cols_add)
            weights =  (new_bonds['Балансовая стоимость пакета(руб.)'] + new_bonds['Переоценка'])/(new_bonds['Балансовая стоимость пакета(руб.)'] + new_bonds['Переоценка']).sum()
    else:
        weights = (bonds['Балансовая стоимость пакета(руб.)'] + bonds['Переоценка'])/(bonds['Балансовая стоимость пакета(руб.)'] + bonds['Переоценка']).sum()
        Data_modDur = Data_modDur
        Data_ytm = Data_ytm[1:253]
    #     Data_ytm = Data_ytm.tail(252)
    
    covMatrix = Data_ytm.pct_change().cov()
    sigma = np.sqrt(np.matmul(np.matmul(weights.T, covMatrix), weights))

   
    mdur_last = np.matmul(weights, (Data_modDur[-1:]).squeeze(axis=0))
    ytm_last = np.matmul(weights, Data_ytm[-1:].squeeze(axis=0))
    
    VaR = mdur_last * ytm_last/100 * norm.ppf(1-Confidence_Interval) * sigma
    
    return VaR   
    

modDurBondsVaR = pd.DataFrame(index = close_pc.index)
modDurBondsVaR['portfolioReturn'] = close_pc['portfolioReturn']
modDurBondsVaR['Mod dur VaR'] = ''
modDurBondsVaR = modDurBondsVaR[252:]


for count, i in tqdm(enumerate(modDurBondsVaR.index)):
    data_ytm_temp = ytm[count : count +253]
    data_mdur_temp = modDur[count: count +252]
    modDurBondsVaR.loc[i, 'Mod dur VaR'] = ModDurVaR(i, data_mdur_temp, data_ytm_temp)
    
    
# modDurBondsVaR['num'] = np.arange(len(modDurBondsVaR))
modDurBondsVaR.plot()

# посчитал для этой и последующих дат вручную
# count = 1562
# i = '2021-10-06 00:00:00'
# ind = i
# Data_ytm = data_ytm_temp
# Data_modDur = data_mdur_temp
# modDurBondsVaR.loc['2021-03-23 00:00:00', 'Mod dur VaR'] = -0.0008054873147806657
# modDurBondsVaR.loc['2021-03-24 00:00:00', 'Mod dur VaR'] = -0.0008054873147806657
# modDurBondsVaR.loc['2021-03-25 00:00:00', 'Mod dur VaR'] = -0.0008054873147806657

   
# modDurBondsVaR[modDurBondsVaR.index > '2020-09-01 00:00:00'][['portfolioReturn', 'Mod dur VaR']].plot()

# data = modDurBondsVaR.copy()
# data = data.rename(columns = {'Mod dur VaR': 'Value-at-Risk', "portfolioReturn": "Доходность"})
# data[['Доходность', "Value-at-Risk"]].plot(title = 'Value-at-Risk', xlabel = 'Дата', ylabel='Доходность')


modDurBondsVaR_Last_Year = modDurBondsVaR.tail(252)
modDurBondsVaR_Last_Year = modDurBondsVaR_Last_Year.rename(columns = {'Mod dur VaR': 'Value-at-Risk', "portfolioReturn": "Доходность"})
modDurBondsVaR_Last_Year.plot(title = 'Value-at-Risk', xlabel = 'Дата', ylabel='Доходность')
# =========================================================================
# Backtesting
# =========================================================================

    
def KupeicPOF(FailureRatio,  NumberOfFailures, Total, Confidence_Interval = 0.01):
    
    """ Calculates Kupiec Proportion of Failure test
    """
    
    temp1 = ((1 - FailureRatio)/(1-Confidence_Interval)) ** (Total - NumberOfFailures)
    temp2 = (FailureRatio/Confidence_Interval) ** NumberOfFailures
    pof = 2 * math.log(temp1 * temp2)
    return pof

def VaRBacktesting(Data, Confidence_Interval = 0.01):
    
    """Backtestinf of Value-at-risk models 
    """   

    breaks = (np.where(Data['portfolioReturn'] < Data['Mod dur VaR'], 1, 0)).sum()
    length = len(Data)
    fr = round(breaks/length, 5)
    z_score = (breaks - Confidence_Interval * length)/np.sqrt(Confidence_Interval * (1-Confidence_Interval)*length)
    p_value_z = round(2*norm.cdf(-z_score), 4)
    pof = KupeicPOF(fr, breaks, length)    
    p_value_pof = round(1 - chi2.cdf(pof, 1), 4)           
    return breaks, fr, p_value_z, p_value_pof
    

breaks, fr, p_value_z, p_value_pof = VaRBacktesting(modDurBondsVaR_Last_Year)

   


    

modDurVar_rubles = modDurBondsVaR['Mod dur VaR'].tail(1) * bonds_portf * -1/1000
modDurVar_rubles_week = modDurVar_rubles * np.sqrt(5)
modDurVar_rubles_month = modDurVar_rubles * np.sqrt(22)

writer = pd.ExcelWriter('bondsVaR ModDur.xlsx', engine='xlsxwriter')
modDurBondsVaR.to_excel(writer, sheet_name = 'all')
modDurBondsVaR_Last_Year.to_excel(writer, sheet_name = 'last year')
Var_in_per_day = ((modDurVar_rubles*1000) / bonds_portf)*100
Var_in_per_month =  ((modDurVar_rubles_month*1000) / bonds_portf)*100
Var_in_per_week =  ((modDurVar_rubles_week*1000) / bonds_portf)*100


writer.save()

del i, k, col, count


