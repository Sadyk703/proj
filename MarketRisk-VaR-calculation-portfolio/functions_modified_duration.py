# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:26:37 2021

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


# ======================================================
# API MOEX authentification
# ======================================================

session = re.Session()


# Authorization ======================================
login = "Sodik.Umurzakov@banksoyuz.ru"
password = input("input the password: ")
 

# autentification ======================================
session.get('https://passport.moex.com/authenticate', auth=(login, password))

# cookies forming ======================================
cookies = {'MicexPassportCert': session.cookies['MicexPassportCert']}


# =======================================================================
# API moex functions
# =======================================================================

def api_query(engine, market, session, secur, from_start, till_end, start):
    """
    the function  froms the url for query
    arguments: 
        engine:     stock
        market:     bonds
        secur:      isin or secid
        from_start: date start
        till_end:   date end
    """
    # param = 'https://iss.moex.com/iss/history/engines/{}/markets/{}/sessions/{}/securities/security.json?q={}&from={}&till={}'.format(engine, market, session, secur, from_start, till_end)
    param = 'https://iss.moex.com/iss/history/engines/{}/markets/{}/sessions/session/securities/{}.json?from={}&till={}&interval=1&start={}'.format(engine, market,  secur, from_start, till_end, start)

    return param


def parse(data, dictor):
    """
    parse data from json format
    """
    columns = data[dictor]['columns']
    data1 = np.array(data[dictor]['data'])
    # metadata = data[dictor]['metadata']
    d = {}

    for i, column in enumerate(columns):
        d[column] = data1[:,i]
    
    return pd.DataFrame(d)

def colnames(engine, market):
    """
    get colnames and their description in Russian
    """
    url = 'https://iss.moex.com/iss/history/engines/{}/markets/{}/securities/columns.json'.format(engine, market)
    return url


def number_query(x):
    """
    defines number of queries denemnding on the total number of rows 
    """    
    if x > 100:
        n =x // 100 +1 
    else:
        n = 1
    return n

def get_data(secur, isin, date_start, date_end):
    """
    get all data from moex api
    """
    session = re.Session()
    
    start = 0
    
    # get url of query
    url = api_query('stock', secur, session, isin, date_start, date_end, start=start)
    
    # get response from server
    response = re.get(url, cookies = cookies)
    
    # get metadata from 'hostory.cursor' dictionary
    df_cursor = parse(response.json(), 'history.cursor')
    
    # get number of queries to send to get whole data. MOEX API allows to download 100 ods only
    number = number_query(df_cursor.loc[0, 'TOTAL'])
    
    # parse data from json format from 'history' dictionary
    data = parse(response.json(), 'history')
    
    if number > 1:
        for i in range(1, number):
            time.sleep(2)
            start += 100
            url = api_query('stock', secur, session, isin, date_start, date_end, start=start)
            response = re.get(url, cookies = cookies)
            df_cursor = parse(response.json(), 'history.cursor')
            df = parse(response.json(), 'history')
            data = pd.concat([data, df])
    
    return data

def SharesDownload(File, date_end):
    """ download close price by ticker from MOEX by api
    """
    
    session = re.Session()
    with open(File, "r") as TICKs:
        TICKs = [line.rstrip() for line in TICKs]
        
    board = 'TQBR'
    shares = pd.DataFrame(apimoex.get_board_history(session, 'GAZA', board=board))
    shares = shares.set_index('TRADEDATE')
    shares = pd.DataFrame(index = shares.index)    
    
    error_shares = []
    
    with re.Session() as session:
        for TICK in TICKs:
             data = apimoex.get_board_history(session, TICK, board=board)
             if data == []:
                 error_shares.append(TICK)
                 continue
             df = pd.DataFrame(data)
             df = df.set_index('TRADEDATE')
             shares = shares.join(df['CLOSE'])
             shares = shares.rename(columns = {'CLOSE': TICK})
    shares = shares[shares.index <= date_end]
    return shares 
  

def BondsDownload(File, date_end):
    """
     downloads data from api moex
     input: File of txt extension with ISINs
            end_date: end_date (YYYY-MM-DD)
            
     output: data - data with close price
             done - isins that had been downloaded
             error - list of isins that hadn't been dowloanded'
     """
    session = re.Session()
  
    # data1 = get_data('bonds', 'RU000A1002L3', '2021-07-16', date_end)

    
    data1 = get_data('bonds', 'RU000A0JTM51', '2021-07-16', date_end)
    data1 = data1.set_index('TRADEDATE')
    
    close_portf = pd.DataFrame(index = data1.index, data=None)
    ytm_portf  = pd.DataFrame(index = data1.index, data=None)
    duration_portf  = pd.DataFrame(index = data1.index, data=None)
    
    done = []
    error = []
    
    with open(File, 'r') as isin:
        ISINs = [line.rstrip() for line in isin]
     
    
    for isin in ISINs:
     
        try:
            data = get_data('bonds', isin, '2021-07-01', date_end)
            data = data.set_index('TRADEDATE')
        
            # close price composition
            close_portf = close_portf.join(data['CLOSE'])
            close_portf = close_portf.rename(columns={"CLOSE": isin})
            
            # ytm composition
            ytm_portf = ytm_portf.join(data['YIELDCLOSE'])
            ytm_portf = ytm_portf.rename(columns = {'YIELDCLOSE': isin})
            
            # # ytm_offer_composition
            # ytm_offer_portf = ytm_offer_portf.join(data['YIELDTOOFFER'])
            # ytm_offer_portf = ytm_offer_portf.rename(columns={'YIELDTOOFFER': isin})
            
            # # duration composition
            duration_portf = duration_portf.join(data['DURATION'])
            duration_portf = duration_portf.rename(columns={'DURATION': isin})
            
            done.append(isin)
        except:
            error.append(isin)
        return  close_portf, error, done

# close_portf.to_excel('close_port.xlsx')  
# ytm_portf.to_excel('ytm_portf.xlsx')
# duration_portf.to_excel('duration_portf.xlsx')


# ======================================================================
# VaR calculation functions
# ======================================================================

def DelNa(Data):
    """ drops rows of df where all values are any
    """
    return Data.dropna(how='all')



def return_calc(ind, data_price, data_volume):
    """
    calculates the return of portfolio depending on the data availabilty
    Input:
        ind - index of dataframe (TRADEDATE expected)
        data_pice - data of close price
        data_volume - data of portfolio volume
    Output:
        portfilio return
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


def FillReturn(Data, DataWeights):
    """
    calculates Portfolio return on th whole data
    Input:
        Data - dataframe
    OutPut:
        Data with filled portfolio returns
    """
    Data = Data.dropna(how='all')
    
    Data['portfolioReturn'] = ''
    for i in Data.index:
        Data.loc[i, 'portfolioReturn'] = return_calc(i, Data, DataWeights)
    Data = Data.sort_index(ascending = False)

    return Data




def VaRCalculation(Data, Formula, Period_Interval,  Confidence_Interval = 0.99, EWMA_lambda = 0.94):
    """ Calculates Value-at-Risk using three methods
        1. Historical simulation
        2. Parametric VaR
        3. Parametric EWMA
    """
    
    # ===================================================
    # Historical simulation
    # ===================================================
    if Formula == 'Historical simulation':
        VaR = np.quantile(Data, 1 - Confidence_Interval)
        return(VaR)
    
    # ===================================================
    # Parametric Normal
    # ===================================================
    if Formula == 'Parametric Normal':
        VaR = Data.mean() - Data.std() * norm.ppf(Confidence_Interval)
        return(VaR)
   
    
    # ===================================================
    # Parametric EWMA
    # ===================================================
    if Formula == 'Parametric EWMA':
        Degree_of_Freedom = np.empty([Period_Interval, ])
        Weights =  np.empty([Period_Interval, ])
        Degree_of_Freedom[0] = 1
        Degree_of_Freedom[1] = EWMA_lambda
        Range = range(Period_Interval)
        for i in range(2,Period_Interval):
            Degree_of_Freedom[i]=Degree_of_Freedom[1]**Range[i]
        for i in range(Period_Interval):
            Weights[i]=Degree_of_Freedom[i]/sum(Degree_of_Freedom)           
        
        sqrdData = Data**2
        EWMAstd = np.sqrt(sum(Weights * sqrdData))
        
        VaR = Data.mean() - EWMAstd * norm.ppf(Confidence_Interval)
        return(VaR)



def FillVaR(Data):
    """
    calculates Vale-at-Risk for the wghole range of data
    Input:
        Data with portfolio returns
    Output:
        Data with portfolio returns and VaRs
    """
    Data = Data.sort_index(ascending = True)
    Data =DelNa(Data)
    # Data = pd.DataFrame(Data)
    
    # Data['Historical VaR'] = ''
    # Data['Parametric VaR'] = ''
    Data['Parametric EWMA'] = ''
   
   
    NewData = Data[252:]
    for count, i in enumerate(NewData.index):
        df = Data[count :count + 252 ][Data.columns[0]]
        df = df.sort_index(ascending=False)
        # NewData.loc[i, 'Historical VaR'] = VaRCalculation(df, Formula = 'Historical simulation')
        # NewData.loc[i, 'Parametric VaR'] = VaRCalculation(df, Formula = 'Parametric Normal')
        NewData.loc[i, 'Parametric EWMA'] = VaRCalculation(df, Formula = 'Parametric EWMA', Period_Interval=252)
        
    # return NewData[[Data.columns[0], 'Historical VaR', 'Parametric VaR', 'Parametric EWMA']]
    final = pd.DataFrame(NewData['Parametric EWMA'])
    # final.columns = Data.columns[0]
    final = final.rename(columns = {'Parametric EWMA' : Data.columns[0]})
    
    return final



# =============================================================================
# New func
# =============================================================================


def FillVaR_shares(Data):
    """
    calculates Vale-at-Risk for the wghole range of data
    Input:
        Data with portfolio returns
    Output:
        Data with portfolio returns and VaRs
    """
    
    Data['Historical VaR'] = ''
    Data['Parametric VaR'] = ''
    Data['Parametric EWMA'] = ''
    
    Data = Data.sort_index(ascending = True)
    
    NewData = Data[252:]
    for count, i in enumerate(NewData.index):
        df = Data[count :count + 252 ]['portfolioReturn']
        df = df.sort_index(ascending=False)
        NewData.loc[i, 'Historical VaR'] = VaRCalculation(df, Period_Interval=252, Formula = 'Historical simulation')
        NewData.loc[i, 'Parametric VaR'] = VaRCalculation(df, Period_Interval=252, Formula = 'Parametric Normal')
        NewData.loc[i, 'Parametric EWMA'] = VaRCalculation(df, Period_Interval=252, Formula = 'Parametric EWMA')
    
    return NewData[['portfolioReturn', 'Historical VaR', 'Parametric VaR', 'Parametric EWMA']]


def KupeicPOF(FailureRatio,  NumberOfFailures, Total, Confidence_Interval = 0.01):
    
    """ Calculates Kupiec Proportion of Failure test
    """
    
    temp1 = ((1 - FailureRatio)/(1-Confidence_Interval)) ** (Total - NumberOfFailures)
    temp2 = (FailureRatio/Confidence_Interval) ** NumberOfFailures
    pof = 2 * math.log(temp1 * temp2)
    return pof


def VaRBacktesting(Formula, Data, Confidence_Interval = 0.01):
    
    """Backtestinf of Value-at-risk models 
    """
    
    if Formula == 'Historical simulation':
        breaks = (np.where(Data['portfolioReturn'] < Data['Historical VaR'], 1, 0)).sum()
        fr = round(breaks/len(Data), 5)
        z_score = (breaks - Confidence_Interval * len(Data))/np.sqrt(Confidence_Interval * (1-Confidence_Interval)*len(Data))
        p_value_z = round(2*norm.cdf(-z_score), 4)
        pof = KupeicPOF(fr, breaks, len(Data))    
        p_value_pof = round(1 - chi2.cdf(pof, 1), 4)           
        return breaks, fr, p_value_z, p_value_pof
    
    if Formula == 'Parametric Normal':
        breaks = (np.where(Data['portfolioReturn'] < Data['Parametric VaR'], 1, 0)).sum()
        fr = round(breaks/len(Data), 5)
        z_score = (breaks - Confidence_Interval * len(Data))/np.sqrt(Confidence_Interval * (1-Confidence_Interval)*len(Data))
        p_value_z = round(2*norm.cdf(-z_score), 4)
        pof = KupeicPOF(fr, breaks, len(Data))    
        p_value_pof = round(1 - chi2.cdf(pof, 1), 4)           
        return breaks, fr, p_value_z, p_value_pof
    
    if Formula == 'Parametric EWMA':
        breaks = (np.where(Data['portfolioReturn'] < Data['Parametric EWMA'], 1, 0)).sum()
        fr = round(breaks/len(Data), 5)
        z_score = (breaks - Confidence_Interval * len(Data))/np.sqrt(Confidence_Interval * (1-Confidence_Interval)*len(Data))
        p_value_z = round(2*norm.cdf(-z_score), 4)
        pof = KupeicPOF(fr, breaks, len(Data))    
        p_value_pof = round(1 - chi2.cdf(pof, 1), 4)           
        return breaks, fr, p_value_z, p_value_pof


def outputBacktest(Data):
    """
    Calculates results of backtesting
    Input:
        Data - data with portfolioReturn and VaR calculated by diff methods
    Ouput: 
        output - data with failure ratio and other statistics
    """
    
    output =  pd.DataFrame(index=['Number of observation', 'Number of exception', "Failure Ratio", 'p_value_Zscore', 'p_valuePOF'],
                               columns = ['Historical simulation', 'Parametric Normal', 'Parametric EWMA'], data = None)
    
    for i in output.index:
        for j in output.columns:
            if i == 'Number of observation':
                output.loc[i, j] = len(Data)
            elif i == 'Number of exception':
                output.loc[i, j] = VaRBacktesting(str(j), Data)[0]
            elif i == "Failure Ratio":
                output.loc[i, j] = VaRBacktesting(str(j), Data)[1]
            elif i == 'p_value_Zscore':
                output.loc[i, j] = VaRBacktesting(str(j), Data)[2]
            elif i == 'p_valuePOF':
                output.loc[i, j] = VaRBacktesting(str(j), Data)[3]
    
    return output
    

def  outputVaR(Data, Volume):
    """
    calculates VaR in rubles
    Input:
        Data - Data with VaR in percents
        Volume - portfolio volume
    Output:
        outputVaR - df with 1-day, weekly and  monthly VaR
    """
    
    outputVaR = pd.DataFrame(index = ['VaR 1-day', 'VaR week', 'VaR month'], columns=['Historical VaR', 'Parametric VaR', 'Parametric EWMA'], data = None)
    Degree = [1, 5, 22]
    for count, i in enumerate(outputVaR.index):
        for col in outputVaR.columns:
            outputVaR.loc[i, col] = round(float(Data.tail(1)[col] * Volume * (-1) * np.sqrt(Degree[count])/1000))
    return outputVaR












