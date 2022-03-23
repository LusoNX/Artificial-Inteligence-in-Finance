import pandas as pd
import yfinance as yf
import investpy
import numpy as np
import sqlite3
import urllib
import pyodbc
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
import time
import statistics



conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=YOUR DIRECTORY\ClusteringDataBase.accdb;')

conn_str_2 =pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=YOUR DIRECTORY\ClusteringDataBase.accdb;')

cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except OSError as err:
        print(f"Error: '{err}'")


def get_fund_holdings(symbol,ID):
    fund = yf.Ticker(symbol)
    info = fund.info
    isin = fund.isin
    exchange = info["exchange"]

    update_values = """
    UPDATE FundIndex
    SET
        ISIN= '{}',
        EXCHANGE = '{}'
    WHERE
        ID = {};
    """.format(isin,exchange,ID)

    execute_query(conn_str_2,update_values)


    # Get price Data
    df_fund = fund.history(period = "max")
    df_fund = df_fund[["Open", "High","Low","Close"]]
    df_fund.reset_index(inplace = True)
    df_fund.rename(columns = {"Date":"Data"},inplace = True)
    df_fund["ID"] = int(ID)
    df_fund.set_index("ID",inplace = True)
    df_fund.to_sql("PriceIndex",acc_engine, if_exists = "append")

    # General Information
    #start_date = info["startDate"]
    df_isin = pd.DataFrame([isin],columns = ["ISIN"])
    #df_isin.to_sql("SELECT ISIN FROM FundIndex WHERE Symbol_yahoo = '{}'".format(symbol),acc_engine, if_exists="replace")


    # Class weights
    list_holdings = ["bondPosition","stockPosition","cashPosition","convertiblePosition","preferredPosition"]
    bond_holdings = info["bondPosition"]
    stock_holdings = info["stockPosition"]
    cash_holdings = info["cashPosition"]
    convertible_holdings = info["convertiblePosition"]
    preferred_holdings = info["preferredPosition"]
    values_holdings = [bond_holdings,stock_holdings,cash_holdings,convertible_holdings,preferred_holdings]
    values_holdings = np.array(values_holdings).reshape(1,len(values_holdings))
    df_holdings = pd.DataFrame(values_holdings, columns = list_holdings)
    df_holdings["ID"] = int(ID)
    df_holdings.set_index("ID",inplace = True)
    df_holdings.to_sql("HoldingIndex",acc_engine,if_exists = "append")

    ## Sector weights (Stock)
    sector_info = info["sectorWeightings"]
    sector_keys_list =[]
    sector_weights_list = []
    for i in sector_info:
        sector_key =list(i.keys())[0]
        sector_weights = list(i.values())[0]
        sector_keys_list.append(sector_key)
        sector_weights_list.append(sector_weights)

    sector_weights_list = np.array(sector_weights_list).reshape(1,len(sector_weights_list))
    df_sector_holdings = pd.DataFrame(sector_weights_list,columns = sector_keys_list)
    df_sector_holdings["ID"] = int(ID)
    df_sector_holdings.set_index("ID",inplace = True)
    df_sector_holdings.to_sql("SectorWeightsIndex",acc_engine,if_exists = "append")

    return df_fund



def get_beta_nav(symbol):
    fund = yf.Ticker(symbol)
    try:
        beta = fund.info["beta3Year"]
    except:
        beta = None

    try:
        nav = fund.info["totalAssets"]
    except:
        nav = None
    return beta,nav

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["returns"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

def max_dd_df(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    draw_dawn_list = []
    for i in range(1,6):
        if i == 1:
            df_1 = df.iloc[0:52*(i)]
            df_1["cum_return"] = (1 + df_1["returns"]).cumprod()
            df_1["cum_roll_max"] = df_1["cum_return"].cummax()
            df_1["drawdown"] = df_1["cum_roll_max"] - df_1["cum_return"]
            df_1["drawdown_pct"] = df_1["drawdown"]/df_1["cum_roll_max"]
            max_dd = df_1["drawdown_pct"].max()
            draw_dawn_list.append(max_dd)
        else:
            value = (52*i-52)
            value_2 = 52*i
            df_2 = df.iloc[value:value_2]
            df_2["cum_return"] = (1 + df_2["returns"]).cumprod()
            df_2["cum_roll_max"] = df_2["cum_return"].cummax()
            df_2["drawdown"] = df_2["cum_roll_max"] - df_2["cum_return"]
            df_2["drawdown_pct"] = df_2["drawdown"]/df_2["cum_roll_max"]
            max_dd = df_2["drawdown_pct"].max()
            draw_dawn_list.append(max_dd)

    max_dd_average = statistics.mean(draw_dawn_list)
    max_dd_average = abs(max_dd_average)
    return max_dd_average


def get_annual_returns(df):
    returns_list = []
    for i in range(1,6):
        if i == 1:
            df_1 = df.iloc[0:52*(i)]
            returns = (df_1.iloc[-1]["Close"] / df_1.iloc[0]["Close"])-1
            returns_list.append(returns)
        else:
            value = (52*i-52)
            value_2 = 52*i
            df_2 = df.iloc[value:value_2]
            returns = (df_2.iloc[-1]["Close"] / df_2.iloc[0]["Close"])-1
            returns_list.append(returns)

    return returns_list


def CAGR(DF,periods):
    df = DF.copy()
    CAGR = (df["Close"].iloc[-1] /df["Close"].iloc[0])**(1/(periods)) -1
    return CAGR



def estimate_stats(_id):
    df_price = pd.read_sql("SELECT * FROM PriceIndex WHERE ID = {}".format(_id),acc_engine)
    symbol = pd.read_sql("SELECT Symbol FROM FundIndex WHERE ID = {}".format(_id),acc_engine).values[0][0]
    df_price["Data"] = pd.to_datetime(df_price["Data"],format = "%Y-%m-%d")
    df_price = df_price.sort_values("Data")
    df_price.set_index("Data",inplace = True)
    df_price_year = df_price.resample("Y").last()

    age = len(df_price_year)

    if age >=5:

    # Get Year return  250 trading days 
        df_price_1_year = df_price[-250:]
        df_price_1_year = df_price_1_year.resample("W").last()
        df_price_1_year["returns"] = df_price_1_year["Close"].pct_change()
        df_price_5_year = df_price[-250*5:]
        df_price_5_year = df_price_5_year.resample("W").last()
        df_price_5_year["returns"] = df_price_5_year["Close"].pct_change()    

        # -----------------------------------------Risk Metrics-----------------------------------------
        #std
        std_1_year = df_price_1_year["returns"][1::].std()*(np.sqrt(52))
        std_5_year = df_price_5_year["returns"][1::].std()*(np.sqrt(52))
        skew_5_year = df_price_5_year["returns"][1::].skew()
        std_downside_1_year = df_price_1_year[df_price_1_year["returns"] <= 0]["returns"].std()*(np.sqrt(52))
        std_downside_5_year = df_price_5_year[df_price_5_year["returns"] <= 0]["returns"].std()*(np.sqrt(52))

        #Var and CVar
        df_sorted_1_year = df_price_1_year.iloc[1::].sort_values(by = ["returns"])
        one_year_var_5 = df_sorted_1_year["returns"].quantile(0.05)
        one_year_cvar_5 = df_sorted_1_year[df_sorted_1_year['returns'].le(df_sorted_1_year['returns'].quantile(0.05))]["returns"].mean()
        df_sorted_5_year = df_price_5_year.iloc[1::].sort_values(by = ["returns"])
        five_year_var_5 = df_sorted_5_year["returns"].quantile(0.05)
        five_year_cvar_5 = df_price_5_year[df_price_5_year['returns'].le(df_price_5_year['returns'].quantile(0.05))]["returns"].mean()
        #Max_DD
        max_dd_1_year = max_dd(df_price_1_year)
        max_dd_5_year = max_dd(df_price_5_year)

        # -----------------------------------------Returns_metrics-----------------------------------------
        cagr_1 = CAGR(df_price_1_year,1)
        cagr_5 = CAGR(df_price_5_year,5)
        
        one_year_tail_gain_5 = df_sorted_1_year["returns"].quantile(0.95)

        one_year_expected_tail_gain_5 = df_sorted_1_year[df_sorted_1_year['returns'] >= df_sorted_1_year["returns"].quantile(0.95)]["returns"].mean()
        five_year_tail_gain_5 = df_sorted_5_year["returns"].quantile(0.95)
        five_year_expected_tail_gain_5 = df_price_5_year[df_price_5_year['returns'] >= df_sorted_1_year["returns"].quantile(0.95)]["returns"].mean()
        


        # Ratio Information 

        # 5 Year Sharpe and Sortino

        annual_returns = get_annual_returns(df_price_5_year)

        sharpe_ratio_5_year = (statistics.mean(annual_returns) /std_5_year)
        sortino_ratio_5_year = (statistics.mean(annual_returns)/std_downside_5_year)

        # 1 Year Sharpe and Sortino
        sharpe_ratio_1_year = annual_returns[0] /std_1_year
        sortino_ratio_1_year = (annual_returns[0]/std_downside_1_year)

            # 5 Year Sterling Ratio
        max_dd_mean = max_dd_df(df_price_5_year)
        sterling_ratio = (statistics.mean(annual_returns))/max_dd_mean

        # Beta Market 
        beta,nav = get_beta_nav(symbol)

        list_of_values_risk = [_id,skew_5_year,std_1_year,std_5_year,std_downside_1_year,std_downside_5_year,one_year_var_5,one_year_cvar_5,five_year_var_5,five_year_cvar_5,max_dd_1_year,max_dd_5_year,beta]
        list_of_values_performance = [_id,nav,age,cagr_1,cagr_5,one_year_tail_gain_5,one_year_expected_tail_gain_5,five_year_tail_gain_5,five_year_expected_tail_gain_5,sharpe_ratio_1_year,sharpe_ratio_5_year,sortino_ratio_1_year,sortino_ratio_5_year,sterling_ratio]

        df_risk_index = pd.read_sql("RiskIndex",acc_engine)
        columns_list_risk = np.array(list(df_risk_index.columns)).reshape(1,len(df_risk_index.columns))
        df_performance_index = pd.read_sql("PerformanceIndex",acc_engine)
        columns_list_performance = np.array(list(df_performance_index.columns)).reshape(1,len(df_performance_index.columns))

        
        df_risk = pd.DataFrame(list_of_values_risk)
        df_risk = df_risk.T
        df_risk.columns = columns_list_risk[0]
        df_risk.set_index("ID",inplace = True)
        df_risk.to_sql("RiskIndex",acc_engine,if_exists = "append")


        df_performance = pd.DataFrame(list_of_values_performance)
        df_performance = df_performance.T
        df_performance.columns = columns_list_performance[0]
        df_performance.set_index("ID",inplace = True)
        df_performance.to_sql("PerformanceIndex",acc_engine,if_exists = "append")

    else:
        print("No Data for fund ID : {}".format(_id))
        pass


def main():
    df_index = pd.read_sql("FundIndex",acc_engine)
    for i,x  in zip(df_index["Symbol"].iloc[646:len(df_index)],df_index["ID"].iloc[646:len(df_index)]):
        try:
            get_fund_holdings(i,x)
            estimate_stats(x)
            print("Sucess for fund {}".format(i))
        except:
            print("Failure for fund {}".format(i))
            pass
        time.sleep(2)


main()
