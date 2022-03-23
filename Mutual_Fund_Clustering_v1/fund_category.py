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
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py


conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
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



def transform_data(df):
    # normal values
    dataset = df.values
    dataset = dataset.reshape(-1,1)
    return dataset


def query_id(_id,conn_str_2):

    df_id = pd.read_sql("SELECT ID FROM ClassifierIndex WHERE ID = {}".format(_id) ,acc_engine)
    if df_id.empty:

        create_id = """
        INSERT INTO  ClassifierIndex(
        ID) VALUES ({})
        """.format(_id)
        execute_query(conn_str_2,create_id)
    else:
        pass


def update_primary_class(_id,primary_class,_conn_str_2):
    conn = _conn_str_2
    update_primary = """
    UPDATE ClassifierIndex
    SET
        ID= {},
        Primary_Classification = '{}'
    WHERE
        ID = {};
    """.format(_id,primary_class,_id)
    execute_query(conn,update_primary)

def update_secondary_class(_id,secondary_class,_conn_str_2):
    conn = _conn_str_2
    update_secondary = """
    UPDATE ClassifierIndex
    SET
        ID= {},
        Second_Classification = '{}'
    WHERE
        ID = {};
    """.format(_id,secondary_class,_id)
    execute_query(conn,update_secondary)

def get_data(_exchange):
    conn_str_2 =pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=YOUR DIRECTORY\ClusteringDataBase.accdb;')
    df_holdings_index = pd.read_sql("HoldingIndex",acc_engine,index_col ="ID")
    df_performance_index = pd.read_sql("PerformanceIndex",acc_engine,index_col ="ID")
    df_risk_index = pd.read_sql("RiskIndex",acc_engine,index_col ="ID")
    df_fund_index = pd.read_sql("FundIndex",acc_engine,index_col ="ID")
    df_sector_weight = pd.read_sql("SectorWeightsIndex",acc_engine,index_col ="ID")
    fund_primary_class_labels = ["Equity Oriented","Fixed Income Oriented", "Balanced Equity_FI", "Special Fund"]
    fund_secondary_class_labels_equity = ["Cyclical Oriented","Sensitive Oriented","Defensive Oriented", "Balanced Sector"]

    fund_primary_class = []
    fund_secondary_class = []
    id_list = []

    for i,x in zip(range(0,len(df_holdings_index)),df_holdings_index.index):
        _id = x
        id_list.append(_id)
        values = df_holdings_index.iloc[i]
        bond_holdings =values["bondPosition"]
        stock_holdings = values["stockPosition"]
        cash_holdings = values["cashPosition"]
        convertible_holdings = values["convertiblePosition"]
        preferred_holdings = values["preferredPosition"]
        special_holdings = preferred_holdings+convertible_holdings

        if stock_holdings >= 0.7:
            fund_primary_class.append(fund_primary_class_labels[0])
            df_sector_weight_s = df_sector_weight[df_sector_weight.index ==_id]
            cyclical_holdings = float(df_sector_weight_s["realestate"].values[0]) +float(df_sector_weight_s["consumer_cyclical"].values[0])+float(df_sector_weight_s["basic_materials"].values[0]) +float(df_sector_weight_s["financial_services"].values[0])
            sensitive_holdings = df_sector_weight_s["technology"].values[0] + df_sector_weight_s["industrials"].values[0] + df_sector_weight_s["communication_services"].values[0]+df_sector_weight_s["energy"].values[0]
            defensive_holdings = 1-sensitive_holdings - cyclical_holdings
            query_id(_id,conn_str_2)
            update_primary_class(_id,fund_primary_class_labels[0],conn_str_2) # Updates SQL Database

            if cyclical_holdings >= 0.5:
                fund_secondary_class.append(fund_secondary_class_labels_equity[0])
                update_secondary_class(_id,fund_secondary_class_labels_equity[0],conn_str_2)

            elif sensitive_holdings >= 0.5:
                fund_secondary_class.append(fund_secondary_class_labels_equity[1])
                update_secondary_class(_id,fund_secondary_class_labels_equity[1],conn_str_2)
            elif defensive_holdings >= 0.5:
                fund_secondary_class.append(fund_secondary_class_labels_equity[2])
                update_secondary_class(_id,fund_secondary_class_labels_equity[2],conn_str_2)
            else:
                fund_secondary_class.append(fund_secondary_class_labels_equity[3])
                update_secondary_class(_id,fund_secondary_class_labels_equity[3],conn_str_2)

        elif bond_holdings >= 0.7:
            fund_primary_class.append(fund_primary_class_labels[1])
            query_id(_id,conn_str_2)
            update_primary_class(_id,fund_primary_class_labels[1],conn_str_2)

        elif special_holdings >= 0.7:
            fund_primary_class.append(fund_primary_class_labels[3])
            query_id(_id,conn_str_2)
            update_primary_class(_id,fund_primary_class_labels[3],conn_str_2)

        else:
            fund_primary_class.append(fund_primary_class_labels[2])
            query_id(_id,conn_str_2)
            update_primary_class(_id,fund_primary_class_labels[2],conn_str_2)





get_data("NAS")








