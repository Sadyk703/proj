# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:19:09 2021

@author: UmurzakovSI
"""

import requests as re
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2
import math
# import apimoex
import time
import subprocess
from tqdm import tqdm

# wd = str(input("Input the working directory: "))
# os.chdir(wd)

os.chdir(r'N:\_ЦУРКР_\2. Финансовые рынки\2. VaR\3. Расчеты\2610')

# ================================================
# Подгружаем данные и создаем датафрейм с котировками облигаций
# ================================================

bonds_close = pd.read_excel('close_portf.xlsx', sheet_name = 'Sheet1')
bonds_close = bonds_close.set_index('TRADEDATE')
bonds_close = bonds_close.fillna(method = 'ffill')

bonds = pd.read_excel('securities.xlsx', sheet_name = 'Bonds')
bonds = bonds.set_index('ISIN')

# Считаем доходность бумаг по цене

bonds_close_pc = np.log(bonds_close).diff()
bonds_close_pc = bonds_close_pc.sort_index(ascending=False)
bonds_close_pc = DelNa(bonds_close_pc)
bonds_close_pc = bonds_close_pc.iloc[:504]

# Заполняем путсые значения за прошлые месяцы (если бумага выпустилась "недавно", мы не можем оценить волотилнлость ее доходности за год)
# Моделируем прошлие данные рандомным образом с помощью нормального распределения

def fillNaN(df):
    a = df.values
    m = np.isnan(a) # mask of NaNs
    mu, sigma = df.mean(), df.std()
    a[m] = np.random.normal(mu, sigma, size=m.sum())
    return df

for i in bonds_close_pc.columns:
    fillNaN(bonds_close_pc[i])

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
    

bonds_close_pc['portfolioReturn'] = ''
for i in bonds_close_pc.index:
    bonds_close_pc.loc[i, 'portfolioReturn'] = return_calc(i, bonds_close_pc, bonds)

plt.plot(bonds_close_pc['portfolioReturn'])

# bonds_close_pc['portfolioReturn'].to_excel('PortfolioReturn.xlsx')



# Считаем обьем портфеля и вес каждой бумаги
bonds_portf = (bonds['Балансовая стоимость пакета(руб.)'] + bonds['Переоценка']).sum()
bonds_weights = (bonds['Балансовая стоимость пакета(руб.)'] + bonds['Переоценка'])/ bonds_portf  
bonds_weights.to_excel('weights.xlsx')

# Считаем процентный VaR для первой бумаги    
bondsVaR = FillVaR(pd.DataFrame(bonds_close_pc[bonds_weights.index[0]]))
bondsVaR.plot()

# Считаем процентный VaR для остальных бумаг
for i in tqdm(range(1, len(bonds_weights.index))):
    bondsVaR = bondsVaR.join(FillVaR(pd.DataFrame(bonds_close_pc[bonds_weights.index[i]])))
    
 # Функция считает процентный VaR для всего портфеля    
def var_price(date):
    var_vector = bondsVaR[bondsVaR.index == date].T
    var_vector['weights'] = bonds_weights
    var_vector.columns = ['Parametric EWMA', 'weights']
    var_vector['estimate'] = var_vector['Parametric EWMA'] * var_vector['weights']
    var_vector = var_vector['estimate']
    var_vector = pd.DataFrame(var_vector)
    var_vector = var_vector[var_vector != 0].dropna()    
    corr_ = bonds_close_pc[var_vector.index[0]].iloc[:252]
    corr_ = pd.DataFrame(corr_)
    for i in var_vector.index:
        corr_[i] = bonds_close_pc[i]
    corr = corr_.corr()
    VAR_final = np.sqrt(np.matmul(np.matmul(var_vector['estimate'].T, corr), var_vector['estimate']))
    VAR_final

    return(VAR_final)

# Считаем VaR портфеля на последнюю дату
var = var_price(bondsVaR.index[251])

# Считаем VaR портфеля на весь период 
VaR_david = pd.DataFrame(index = bondsVaR.index, columns = ['VaR'])       
for i in tqdm(VaR_david.index):
    VaR_david.loc[i, 'VaR'] = var_price(i)

VaR_david.sort_index(ascending = False, inplace = True)

# VaR в рублях на 1 день
Value_at_Risk = var * bonds_portf
Value_at_Risk


# var2 = var_price(bondsVaR.index[250])
# Value_at_Risk2 = var2 * bonds_portf


# var3 = var_price(bondsVaR.index[249])
# Value_at_Risk3 = var3 * bonds_portf


# var4 = var_price(bondsVaR.index[248])
# Value_at_Risk4 = var4 * bonds_portf


# var5 = var_price(bondsVaR.index[247])
# Value_at_Risk5 = var5 * bonds_portf

# var6 = var_price(bondsVaR.index[246])
# Value_at_Risk6 = var6 * bonds_portf




bondsVaR.to_excel('bondsVaR.xlsx')

var_price(bondsVaR.index[246])


Value_at_Risk_week = Value_at_Risk * np.sqrt(5)
Value_at_Risk_month = Value_at_Risk * np.sqrt(22)

Var_in_per_day =  ( Value_at_Risk/ bonds_portf)*100
Var_in_per_month =  ( Value_at_Risk_month/ bonds_portf)*100
Var_in_per_week =  (Value_at_Risk_week / bonds_portf)*100

# =============================================================================
# Sodik's calculation of VaR
var1 = FillVaR_shares(bonds_close_pc)
var1.plot()

BondVaR_LastYear = var1.tail(251)
 
# shares data Output
# sharesBack = outputBacktest(BondVaR_LastYear)

# BondVaR_LastYear = BondVaR_LastYear.rename(columns = {'portfolioReturn': "Доходность", "Historical VaR": "Исторический метод",
#                                                         "Parametric VaR": "Параметрический метод", "Parametric EWMA": "Параметрический EWMA"})


BondsVaR_rubles = outputVaR(BondVaR_LastYear, bonds_portf)
BondsVaR_rubles['Объем портфеля'] = bonds_portf/1000
BondsVaR_rubles['Исторический метод, %'] = (BondsVaR_rubles['Historical VaR']/BondsVaR_rubles['Объем портфеля'])*100
BondsVaR_rubles['Параметрический метод, %'] = (BondsVaR_rubles['Parametric VaR']/BondsVaR_rubles['Объем портфеля'])*100
BondsVaR_rubles['Параметрический EWMA, %'] = (BondsVaR_rubles['Parametric EWMA']/BondsVaR_rubles['Объем портфеля'])*100
BondsVaR_rubles = BondsVaR_rubles.rename(columns = {"Historical VaR": "Исторический метод, тыс. руб.",
                                                        "Parametric VaR": "Параметрический метод, тыс. руб.", "Parametric EWMA": "Параметрический EWMA, тыс. руб."})

BondsVaR_rubles.to_excel('BondsVaR_rubles.xlsx')

# =============================================================================

# ================================================
# shares data section
# ================================================

# get data from moex
# shares_close = SharesDownload('ticks_shares.txt', date)
shares_close = pd.read_excel('shares_close.xlsx')
shares_close = shares_close.set_index('TRADEDATE')
shares_close = shares_close.fillna(method='ffill')

# calculate percent change
shares_close_pc =  shares_close.pct_change()
shares_close_pc = shares_close_pc.sort_index(ascending=True)


shares = pd.read_excel('securities.xlsx', sheet_name = 'Shares')
shares = shares.set_index('Ticker')

shares_torg = shares[shares['Торг'] == 1]

shares_torg_portf = (shares_torg['Балансовая стоимость пакета(руб.)'] + shares_torg['Переоценка']).sum()
shares_weigths = (shares_torg['Балансовая стоимость пакета(руб.)'] + shares_torg['Переоценка']) / shares_torg_portf

shares_close_pc = FillReturn(shares_close_pc, shares_torg)


# ==================================================================================
# Shares VaR calculation section
# ==================================================================================

# calculates VaR of  shares
sharesVaR = FillVaR_shares(shares_close_pc)
sharesVaR.plot()

sharesVaR_LastYear = sharesVaR.tail(252)
 
# shares data Output
# sharesBack = outputBacktest(sharesVaR_LastYear)

sharesVaR_rubles = outputVaR(sharesVaR_LastYear, shares_torg_portf)
sharesVaR_rubles['Объем портфеля'] = shares_torg_portf/1000
sharesVaR_rubles.to_excel('sharesVaR_rubles.xlsx')

sharesVaR_LastYear = sharesVaR_LastYear.rename(columns = {'portfolioReturn': "Доходность", "Historical VaR": "Исторический метод",
                                                        "Parametric VaR": "Параметрический метод", "Parametric EWMA": "Параметрический EWMA"})
sharesVaR_LastYear.plot(title = 'Value-at-Risk Акции', xlabel = 'Дата', ylabel = 'Доходность')

sharesLoss = [ shares_close_pc.loc[i, 'portfolioReturn']* shares_torg_portf for i in shares_close_pc.tail(5).index]



# ==================================================================================
# REPORT
# ==================================================================================

# Создаем датафрейм с базовой информацией
general_info = pd.DataFrame(index = None, columns = ['Объем портфеля, млн. руб.', 'Количество бумаг, шт.', 
                                                     'Дюрация портфеля, лет', 'Доходность портфеля, %']) 

general_info['Объем портфеля, млн. руб.'] = bonds_portf/10**6
general_info['Количество бумаг, шт.'] = len(bonds)
general_info['Доходность портфеля, %'] = bonds_close_pc['portfolioReturn'].head(1)*np.sqrt(252)


plt.plot(bonds_close_pc.index[:252], bonds_close_pc['portfolioReturn'][:252], label = 'Portfolio Return')
plt.plot(bonds_close_pc.index[:252], -VaR_david['VaR'], label = 'Value at Risk')
plt.legend()
plt.savefig('plot.png')
plt.show() 


# Создаем датафрейм с основными риск-метриками
bonds_var_table = pd.DataFrame(index = None, columns = ["Название Эмитента",'Позиция, млн. руб.', 'Вес, %', 'Индивидуальный VaR, %', 
                                                        'Индивидуальный VaR, тыс. руб.', 'Маржинальный VaR, %', 'Компонентный VaR, тыс. руб.']) 
bonds_var_table['Название Эмитента'] = bonds[bonds.columns[1]]
bonds_var_table['Вес, %'] = bonds_weights
pos = []
for i in range(len(bonds_var_table)):
    p = bonds.loc[bonds_var_table.index[i]]
    pos.append(round((p.loc['Балансовая стоимость пакета(руб.)'] + p.loc['Переоценка'])/1000000, 2))    
bonds_var_table['Позиция, млн. руб.'] = pos
bonds_var_table['Индивидуальный VaR, %'] = bondsVaR.tail(1).T
bonds_var_table['Индивидуальный VaR, тыс. руб.'] = bonds_var_table['Позиция, млн. руб.'] * -bonds_var_table['Индивидуальный VaR, %']

# Функция для расчета маржинального VaR
def marginal_var(isin):
    i = bonds_close_pc[isin]
    # p = bonds_close_pc[(bonds_close_pc.loc[:, bonds_close_pc.columns != isin]) & (bonds_close_pc.loc[:, bonds_close_pc.columns != 'portfolioReturn'])]            
    p = bonds_close_pc.loc[:, (bonds_close_pc.columns != isin) & (bonds_close_pc.columns != 'portfolioReturn')] 
    for date in p.index:
        p.loc[date, 'portfolioReturn'] = return_calc(date, bonds_close_pc, bonds)
    cor = i.corr(p['portfolioReturn'])
    std_i = i.std()
    std_p = p['portfolioReturn'].std()
    m_var = cor*std_i/std_p
    return(m_var)

# расчет маржинального VaR
m_vars = []
for i in tqdm(bonds.index):
    m_vars.append(marginal_var(i))
    
bonds_var_table['Маржинальный VaR, %'] = m_vars
bonds_var_table['Компонентный VaR, тыс. руб.'] = Value_at_Risk/1000 * bonds_var_table['Маржинальный VaR, %'] * bonds_var_table['Вес, %']


# Информация по портфелю
portfolio_var = pd.DataFrame(index = [1], columns = ['VaR диверсифицированный, %', 'VaR диверсифицированный, тыс. руб.', 
                                                        'VaR недиверсифицированный, тыс. руб.', 'Эффект диверсификации, тыс. руб.'])
portfolio_var.index = general_info.index
portfolio_var['VaR диверсифицированный, %'] = var
portfolio_var['VaR диверсифицированный, тыс. руб.'] = Value_at_Risk/1000
portfolio_var['VaR недиверсифицированный, тыс. руб.'] = sum(bonds_var_table['Индивидуальный VaR, тыс. руб.'])*1000
portfolio_var['Эффект диверсифицированный, тыс. руб.'] = portfolio_var['VaR диверсифицированный, тыс. руб.'] - portfolio_var['VaR недиверсифицированный, тыс. руб.']

# =============================================================================
# Mod by Sodik
# VaR in percent for shares


# =============================================================================



# Creating Excel Writer Object from Pandas  
writer = pd.ExcelWriter('VaR_report.xlsx',engine='xlsxwriter')   
workbook=writer.book
worksheet=workbook.add_worksheet('ValueAtRisk')
writer.sheets['ValueAtRisk'] = worksheet
general_info.to_excel(writer,sheet_name='ValueAtRisk',startrow=0 , startcol=0)   
portfolio_var.to_excel(writer,sheet_name='ValueAtRisk',startrow=5, startcol=0) 
bonds_var_table.to_excel(writer,sheet_name='ValueAtRisk',startrow=10, startcol=0)
writer.save()





# HTML
# 1. Set up multiple variables to store the titles, text within the report
page_title_text='Отчет по контролю лимитов VaR'
title_text = 'Отчет по контролю лимитов VaR'
text = 'Контактное лицо: Анисимов Н.Д.'
text_2 = 'TEXT TEXT TEXT TEXT TEXT TEXT'
text_3 = 'TEXT TEXT TEXT TEXT TEXT TEXT'


# 2. Combine them together using a long f-string
html = f'''
    <html>
        <head>
            <title>{page_title_text}</title>
        </head>
        <body>
            <h1>{title_text}</h1>
            <p>{text}</p>
            <img src= 'plot.png' alt = 'График доходности и VaR' width="600" height = "400">
            <h2>{text_2}</h2>
            {general_info.to_html()}
            <h2>{text_3}</h2>
            {portfolio_var.to_html()}
            <h3>{text_3}</h3>
            {bonds_var_table.to_html()}
            
        </body>
    </html>
    '''
# 3. Write the html string as an HTML file
with open('html_report.html', 'w') as f:
    f.write(html)






