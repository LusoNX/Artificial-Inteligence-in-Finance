
import pandas as pd
import sys
from sqlite3 import connect
import urllib
from sqlalchemy import create_engine
import pyodbc
import urllib


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except OSError as err:
        print(f"Error: '{err}'")


def createDataBase():
    conn_str = pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=YOUR DIRECOTRY\ClusteringDataBase.accdb;')

    FundIndex_table = """
    CREATE TABLE FundIndex(
    ID INT PRIMARY KEY,
    Symbol VARCHAR(50),
    Fund_name VARCHAR(50),
    ISIN VARCHAR(50),
    EXCHANGE VARCHAR(50)
    );
    """

    PriceIndex_table = """
    CREATE TABLE PriceIndex(
    ID INT,
    Data DATETIME,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT
    );
    """

    HoldingIndex_table = """
    CREATE TABLE HoldingIndex(
    ID INT,
    bondPosition FLOAT,
    stockPosition FLOAT,
    cashPosition FLOAT,
    convertiblePosition FLOAT,
    preferredPosition FLOAT
    );
    """

    SectorWeightIndex_table = """
    CREATE TABLE SectorWeightsIndex(
    ID INT,
    realestate VARCHAR(50),
    consumer_cyclical FLOAT,
    basic_materials FLOAT,
    consumer_defensive FLOAT,
    technology FLOAT,
    communication_services FLOAT,
    financial_services FLOAT,
    utilities FLOAT,
    industrials FLOAT,
    energy FLOAT,
    healthcare FLOAT
    );
    """

    RiskIndex_table = """
    CREATE TABLE RiskIndex(
    ID INT,
    Skew_5_year FLOAT,
    Std_1_Year FLOAT,
    Std_5_year FLOAT,
    Std_1_year_downside FLOAT,
    Std_5_year_downside FLOAT,
    Var_5_1_year FLOAT,
    CVar_5_1_year FLOAT,
    Var_5_5_year FLOAT,
    CVar_5_5_year FLOAT,
    Max_DD_1_year FLOAT,
    Max_DD_5_year FLOAT,
    Beta_Market FLOAT
    );
    """

    PerformanceIndex_table = """
    CREATE TABLE PerformanceIndex(
    ID INT,
    NAV FLOAT,
    Age INT,
    CAGR_1_year FLOAT,
    CAGR_5_year FLOAT,  
    Tail_gain_5_1_year FLOAT,
    Expected_tail_gain_5_1_year FLOAT,
    Tail_gain_5_5_year FLOAT,
    Expected_tail_gain_5_5_year FLOAT,
    Sharpe_Ratio_1_year FLOAT,
    Sharpe_Ratio_5_year FLOAT,
    Sortino_Ratio_1_year FLOAT,
    Sortino_Ratio_5_year FLOAT,
    Starling_5_year FLOAT
    );
    """
    ClassifierIndex_table = """
    CREATE TABLE ClassifierIndex(
    ID INT,
    Primary_Classification VARCHAR(50),
    Second_Classification VARCHAR(50),
    Third_Classification  VARCHAR(50)
    );
    """


    


    execute_query(conn_str,FundIndex_table)
    execute_query(conn_str,PriceIndex_table)
    execute_query(conn_str,HoldingIndex_table)
    execute_query(conn_str,SectorWeightIndex_table)
    execute_query(conn_str,PerformanceIndex_table)
    execute_query(conn_str,RiskIndex_table)
    execute_query(conn_str,ClassifierIndex_table)


    #Populate the dict data

#    dict_data = pd.read_csv("dictionary.csv", sep = ";")
#    dict_data.rename(columns = {"codigo_ms":"MorningStar_code"},inplace = True)
#    dict_data.to_sql("DictIndex",acc_engine,if_exists = "replace")


createDataBase()



## Populate the dict table




