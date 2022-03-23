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




def update_third_class(_id,third_class,_conn_str_2):
    conn = _conn_str_2
    update_third = """
    UPDATE ClassifierIndex
    SET
        ID= {},
        Third_Classification = '{}'
    WHERE
        ID = {};
    """.format(_id,third_class,_id)
    execute_query(conn,update_third)


def transform_data(df):
    # normal values
    dataset = df.values
    dataset = dataset.reshape(-1,1)
    return dataset

def get_data(_exchange,_class_fund,_subclass):
    df_holdings_index = pd.read_sql("HoldingIndex",acc_engine,index_col ="ID")
    df_performance_index = pd.read_sql("PerformanceIndex",acc_engine,index_col ="ID")
    df_risk_index = pd.read_sql("RiskIndex",acc_engine,index_col ="ID")
    df_fund_index = pd.read_sql("FundIndex",acc_engine,index_col ="ID")
    df_sector_weight = pd.read_sql("SectorWeightsIndex",acc_engine,index_col ="ID")
    df_classifier= pd.read_sql("ClassifierIndex",acc_engine,index_col = "ID")
    #df_price_index = pd.read_sql("PriceIndex",acc_engine,index_col = "ID")

    id_exists = list(df_performance_index.index.values)

    #df_holdings_index = df_holdings_index[df_holdings_index.index.isin(id_exists)]
    #df_performance_index = df_performance_index[df_performance_index.index.isin(id_exists)]
    #df_risk_index = df_risk_index[df_risk_index.index.isin(id_exists)]
    #df_fund_index = df_fund_index[df_fund_index.index.isin(id_exists)]
    #df_sector_weight = df_sector_weight[df_sector_weight.index.isin(id_exists)]
    #df_fund_index = df_fund_index[df_fund_index.index.isin(id_exists)]
    #df_price_index = df_price_index[df_price_index.index.isin(id_exists)]


    ## SET THE INDEXES BEFORE SQL
    #df_sector_weight.to_sql("SectorWeightsIndex",acc_engine,if_exists = "replace")
    #df_fund_index.to_sql("FundIndex",acc_engine,if_exists= "replace")
    #df_holdings_index.to_sql("HoldingIndex",acc_engine,if_exists = "replace")
    #df_holdings_index.to_sql("HoldingIndex",acc_engine,if_exists = "replace")
    #df_price_index.to_sql("PriceIndex",acc_engine,if_exists = "replace")

    id_nas = list(df_fund_index[df_fund_index["EXCHANGE"] == _exchange].index.values)
    id_class = list(df_classifier[df_classifier["Primary_Classification"] == _class_fund].index.values)

    df_holdings_index = df_holdings_index[df_holdings_index.index.isin(id_nas)]
    df_performance_index = df_performance_index[df_performance_index.index.isin(id_nas)]
    df_risk_index = df_risk_index[df_risk_index.index.isin(id_nas)]
    df_fund_index = df_fund_index[df_fund_index.index.isin(id_nas)]
    df_sector_weight = df_sector_weight[df_sector_weight.index.isin(id_nas)]

    if _subclass == True:
        df_holdings_index = df_holdings_index[df_holdings_index.index.isin(id_class)]
        df_performance_index = df_performance_index[df_performance_index.index.isin(id_class)]
        df_risk_index = df_risk_index[df_risk_index.index.isin(id_class)]
        df_fund_index = df_fund_index[df_fund_index.index.isin(id_class)]
        df_sector_weight = df_sector_weight[df_sector_weight.index.isin(id_class)]
    else:
        pass

    symbols = pd.DataFrame(df_fund_index["Symbol"], index = df_fund_index.index)
    df_merged = symbols.merge(df_performance_index,left_index = True, right_index = True)
    df_merged = df_merged.merge(df_risk_index, left_index = True,right_index = True)
    df_merged  = df_merged.merge(df_holdings_index, left_index = True, right_index = True)
    return df_merged

def PCA_function_2(df):
    scaler = StandardScaler()

    # Risk factors
    df_risk = df[['Skew_5_year', 'Std_1_Year',
       'Std_5_year', 'Std_1_year_downside', 'Std_5_year_downside',
       'Max_DD_1_year', 'Max_DD_5_year']]

    df_performance = df[['CAGR_1_year', 'CAGR_5_year',
       'Tail_gain_5_1_year', 'Expected_tail_gain_5_1_year',
       'Tail_gain_5_5_year', 'Expected_tail_gain_5_5_year']]



    # Performance Factors
    df_std_risk = scaler.fit_transform(df_risk)
    df_std_performance = scaler.fit_transform(df_performance)


    pca = PCA()
    pca.fit(df_std_performance)
    pca_variance_vectors = pca.explained_variance_ratio_


    plt.figure(figsize = (10,6))
    plt.plot(range(1,7),pca_variance_vectors.cumsum(), marker = "o", linestyle = "--")
    plt.title("Explained Variance by factors")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumultative Explained Variance")
    plt.show()

    pca_2 = PCA()
    pca_2.fit(df_std_risk)
    pca_variance_vectors_2 = pca_2.explained_variance_ratio_
    plt.figure(figsize = (10,6))
    plt.plot(range(1,8),pca_variance_vectors_2.cumsum(), marker = "o", linestyle = "--")
    plt.title("Explained Variance by factors")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumultative Explained Variance")
    plt.show()



    # Both find optimal levels of 95 % at 3 n PCA
    pca = PCA(n_components =3)
    pca_analysis_risk = pca.fit(df_std_risk)
    pca_analysis_performance = pca.fit(df_std_performance)
    pca_final_risk =pca_analysis_risk.fit_transform(df_std_risk)
    pca_final_performance =pca_analysis_performance.fit_transform(df_std_performance)
    return pca_final_risk, pca_final_performance

def PCA_function(df):
    scaler = StandardScaler()

    # Risk factors
    df_risk = df[['Skew_5_year', 'Std_1_Year',
       'Std_5_year', 'Std_1_year_downside', 'Std_5_year_downside',
       'Var_5_1_year', 'CVar_5_1_year', 'Var_5_5_year', 'CVar_5_5_year',
       'Max_DD_1_year', 'Max_DD_5_year']]

    df_performance = df[['CAGR_1_year', 'CAGR_5_year',
       'Tail_gain_5_1_year', 'Expected_tail_gain_5_1_year',
       'Tail_gain_5_5_year', 'Expected_tail_gain_5_5_year',
       'Sharpe_Ratio_1_year', 'Sharpe_Ratio_5_year', 'Sortino_Ratio_1_year',
       'Sortino_Ratio_5_year', 'Starling_5_year']]



    # Performance Factors
    df_std_risk = scaler.fit_transform(df_risk)
    df_std_performance = scaler.fit_transform(df_performance)

    pca = PCA()
    pca.fit(df_std_performance)
    pca_variance_vectors = pca.explained_variance_ratio_

    plt.figure(figsize = (10,6))
    plt.plot(range(1,12),pca_variance_vectors.cumsum(), marker = "o", linestyle = "--")
    plt.title("Explained Variance by factors")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumultative Explained Variance")
    plt.show()

    pca_2 = PCA()
    pca_2.fit(df_std_risk)
    pca_variance_vectors_2 = pca_2.explained_variance_ratio_
    plt.figure(figsize = (10,6))
    plt.plot(range(1,12),pca_variance_vectors_2.cumsum(), marker = "o", linestyle = "--")
    plt.title("Explained Variance by factors")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumultative Explained Variance")
    plt.show()



    # Both find optimal levels of 95 % at 3 n PCA
    pca = PCA(n_components =4)
    pca_analysis_risk = pca.fit(df_std_risk)
    pca_analysis_performance = pca.fit(df_std_performance)
    pca_final_risk =pca_analysis_risk.fit_transform(df_std_risk)
    pca_final_performance =pca_analysis_performance.fit_transform(df_std_performance)
    return pca_final_risk, pca_final_performance




def Cluster_Classifier(stage):
    if stage ==1:
        df_merged = get_data("NAS","Equity Oriented")
        df_merged = df_merged[["Symbol",'bondPosition','stockPosition', 'cashPosition', 'convertiblePosition','preferredPosition']]
        df_merged.set_index("Symbol",inplace = True)

        X = df_merged.values

        sse = []
        for k in range(2,15):
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(data_final)
            sse.append(kmeans.inertia_)



        plt.plot(range(2,15),sse)
        plt.title("Elbow Curve")
        #plt.show()
        kmeans = KMeans(n_clusters =6,random_state = 0).fit(data_final)
        labels = KMeans(n_clusters=6,random_state = 0).fit_predict(data_final)

        sns.countplot(labels).set_title("Cluster Fund Count")
        #plt.show()
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data_final[labels == 0,0],data_final[labels == 0,1],data_final[labels == 0,2], s = 40 , color = 'blue', label = "cluster 0")
        ax.scatter(data_final[labels == 1,0],data_final[labels == 1,1],data_final[labels == 1,2], s = 40 , color = 'orange', label = "cluster 1")
        ax.scatter(data_final[labels == 2,0],data_final[labels == 2,1],data_final[labels == 2,2], s = 40 , color = 'green', label = "cluster 2")
        ax.scatter(data_final[labels == 3,0],data_final[labels == 3,1],data_final[labels == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
        ax.scatter(data_final[labels == 4,0],data_final[labels == 4,1],data_final[labels == 4,2], s = 40 , color = 'purple', label = "cluster 4")
        #ax.set_xlabel('Age of a customer-->')
        #ax.set_ylabel('Anual Income-->')
        #ax.set_zlabel('Spending Score-->')
        ax.legend()
        #plt.show()



        centroids = kmeans.cluster_centers_
        u_labels = np.unique(labels)

        plt.figure(figsize=(20,10))
        for i in u_labels:
            plt.scatter(data_final[labels == i,0], data_final[labels ==i,1], label = i)
        plt.scatter(data_final[:,0],data_final[:,1],s = 50, c = kmeans.labels_, cmap = "rainbow")
        plt.scatter(centroids[:,0], centroids[:,1], s = 50, color = "k",label ="centroids")
        plt.legend()
        #plt.show()


        df_symbol = pd.DataFrame(df_symbol.index)
        cluster_labels = pd.DataFrame(kmeans.labels_)
        df = pd.concat([df_symbol,cluster_labels],axis = 1)


    elif stage ==2:
        df_merged = get_data("NAS","Equity Oriented",True)
        pca_risk, pca_performance = PCA_function(df_merged)
        df_merged = df_merged[["Symbol","Sharpe_Ratio_1_year","Sortino_Ratio_1_year","Starling_5_year"]]
        df_symbol = df_merged[["Symbol"]]

        df_merged.set_index("Symbol",inplace = True)
        df_symbol.set_index("Symbol",inplace = True)
        X = df_merged[["Sharpe_Ratio_1_year","Sortino_Ratio_1_year","Starling_5_year",]].values

        scalar = StandardScaler()
        #ratio_values = transform_data(df_merged[["Sharpe_Ratio_1_year","Sortino_Ratio_1_year","Starling_5_year"]])
        X = scalar.fit_transform(X)
        data_final = X
        #data_final = np.append(X,nav,axis =1)

        sse = []
        for k in range(2,15):
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(data_final)
            sse.append(kmeans.inertia_)

        plt.plot(range(2,15),sse)
        plt.title("Elbow Curve")
        #plt.show()

        kmeans = KMeans(n_clusters =3,random_state = 0).fit(data_final)
        labels = KMeans(n_clusters=3,random_state = 0).fit_predict(data_final)

        print(data_final)

        sns.countplot(labels).set_title("Cluster Fund Count")
        #plt.show()
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data_final[labels == 0,0],data_final[labels == 0,1],data_final[labels == 0,2], s = 40 , color = 'blue', label = "cluster 0")
        ax.scatter(data_final[labels == 1,0],data_final[labels == 1,1],data_final[labels == 1,2], s = 40 , color = 'orange', label = "cluster 1")
        ax.scatter(data_final[labels == 2,0],data_final[labels == 2,1],data_final[labels == 2,2], s = 40 , color = 'green', label = "cluster 2")
        #ax.scatter(data_final[labels == 3,0],data_final[labels == 3,1],data_final[labels == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
        #ax.scatter(data_final[labels == 4,0],data_final[labels == 4,1],data_final[labels == 4,2], s = 40 , color = 'purple', label = "cluster 4")
        #ax.set_xlabel('Age of a customer-->')
        #ax.set_ylabel('Anual Income-->')
        #ax.set_zlabel('Spending Score-->')
        ax.legend()
        #plt.show()


        centroids = kmeans.cluster_centers_
        u_labels = np.unique(labels)
        plt.figure(figsize=(20,10))
        for i in u_labels:
            plt.scatter(data_final[labels == i,0], data_final[labels ==i,1], label = i)
        plt.scatter(data_final[:,0],data_final[:,1],s = 50, c = kmeans.labels_, cmap = "rainbow")
        plt.scatter(centroids[:,0], centroids[:,1], s = 50, color = "k",label ="centroids")
        plt.legend()
        #plt.show()


        df_symbol = pd.DataFrame(df_symbol.index)
        cluster_labels = pd.DataFrame(kmeans.labels_)
        df = pd.concat([df_symbol,cluster_labels],axis = 1)



    elif stage ==3:
        df_merged = get_data("NAS","Equity Oriented",True)
        df_merged.dropna(inplace = True)
        pca_risk, pca_performance = PCA_function(df_merged)

        df_symbol = df_merged[["Symbol"]]
        df_symbol.set_index("Symbol",inplace = True)
        scalar = StandardScaler()
        nav = transform_data(df_merged["NAV"])
        age = transform_data(df_merged["Age"])


        nav_std = scalar.fit_transform(nav)
        age_std = scalar.fit_transform(age)


        #X = np.array(list(df_symbol.index.values)).reshape(-1,1)
        #data_final = np.append(X,pca_risk,axis =1)
        data_final = np.append(pca_risk,pca_performance,axis = 1)
        data_final = np.append(data_final,age_std,axis = 1)
        data_final = np.append(data_final,nav_std,axis =1)


        sse = []
        for k in range(2,15):
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(data_final)
            sse.append(kmeans.inertia_)



        plt.plot(range(2,15),sse)
        plt.title("Elbow Curve")
        plt.show()

        kmeans = KMeans(n_clusters =5,random_state = 0).fit(data_final)
        labels = KMeans(n_clusters=5,random_state = 0).fit_predict(data_final)



        sns.countplot(labels).set_title("Cluster Fund Count")
        plt.show()
        fig = plt.figure(figsize = (10,15))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data_final[labels == 0,0],data_final[labels == 0,1],data_final[labels == 0,2], s = 40 , color = 'blue', label = "cluster 0")
        ax.scatter(data_final[labels == 1,0],data_final[labels == 1,1],data_final[labels == 1,2], s = 40 , color = 'orange', label = "cluster 1")
        ax.scatter(data_final[labels == 2,0],data_final[labels == 2,1],data_final[labels == 2,2], s = 40 , color = 'green', label = "cluster 2")
        ax.scatter(data_final[labels == 3,0],data_final[labels == 3,1],data_final[labels == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
        ax.scatter(data_final[labels == 4,0],data_final[labels == 4,1],data_final[labels == 4,2], s = 40 , color = 'purple', label = "cluster 4")
        #ax.set_xlabel('Age of a customer-->')
        #ax.set_ylabel('Anual Income-->')
        #ax.set_zlabel('Spending Score-->')
        ax.legend()
        plt.show()



        centroids = kmeans.cluster_centers_
        u_labels = np.unique(labels)

        plt.figure(figsize=(20,10))
        for i in u_labels:
            plt.scatter(data_final[labels == i,0], data_final[labels ==i,1], label = i)
        plt.scatter(data_final[:,0],data_final[:,1],s = 50, c = kmeans.labels_, cmap = "rainbow")
        plt.scatter(centroids[:,0], centroids[:,1], s = 50, color = "k",label ="centroids")
        plt.legend()
        plt.show()


        df_symbol = pd.DataFrame(df_symbol.index)
        cluster_labels = pd.DataFrame(kmeans.labels_)
        df = pd.concat([df_symbol,cluster_labels],axis = 1)
        df.rename(columns = {0:"Third_Classification"},inplace = True)
        return df



    elif stage ==4:
        df_merged = get_data("NAS","Equity Oriented",True)
        df_merged.dropna(inplace = True)
        pca_risk, pca_performance = PCA_function_2(df_merged)

        df_symbol = df_merged[["Symbol"]]
        df_symbol.set_index("Symbol",inplace = True)
        scalar = StandardScaler()
        nav = transform_data(df_merged["NAV"])
        age = transform_data(df_merged["Age"])


        nav_std = scalar.fit_transform(nav)
        age_std = scalar.fit_transform(age)
        #X = np.array(list(df_symbol.index.values)).reshape(-1,1)
        #data_final = np.append(X,pca_risk,axis =1)
        data_final = np.append(pca_risk,pca_performance,axis = 1)
        #data_final = np.append(data_final,age_std,axis = 1)
        data_final = np.append(data_final,nav_std,axis =1)


        sse = []
        for k in range(2,15):
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(data_final)
            sse.append(kmeans.inertia_)



        plt.plot(range(2,15),sse)
        plt.title("Elbow Curve")
        plt.show()

        kmeans = KMeans(n_clusters =3,random_state = 0).fit(data_final)
        labels = KMeans(n_clusters=3,random_state = 0).fit_predict(data_final)



        sns.countplot(labels).set_title("Cluster Fund Count")
        plt.show()
        fig = plt.figure(figsize = (10,15))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data_final[labels == 0,0],data_final[labels == 0,1],data_final[labels == 0,2], s = 40 , color = 'blue', label = "cluster 0")
        ax.scatter(data_final[labels == 1,0],data_final[labels == 1,1],data_final[labels == 1,2], s = 40 , color = 'orange', label = "cluster 1")
        ax.scatter(data_final[labels == 2,0],data_final[labels == 2,1],data_final[labels == 2,2], s = 40 , color = 'green', label = "cluster 2")
        #ax.scatter(data_final[labels == 3,0],data_final[labels == 3,1],data_final[labels == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
        #ax.scatter(data_final[labels == 4,0],data_final[labels == 4,1],data_final[labels == 4,2], s = 40 , color = 'purple', label = "cluster 4")
        ax.legend()
        plt.show()



        centroids = kmeans.cluster_centers_
        u_labels = np.unique(labels)

        plt.figure(figsize=(20,10))
        for i in u_labels:
            plt.scatter(data_final[labels == i,0], data_final[labels ==i,1], label = i)
        plt.scatter(data_final[:,0],data_final[:,1],s = 50, c = kmeans.labels_, cmap = "rainbow")
        plt.scatter(centroids[:,0], centroids[:,1], s = 50, color = "k",label ="centroids")
        plt.legend()
        plt.show()


        df_symbol = pd.DataFrame(df_symbol.index)
        cluster_labels = pd.DataFrame(kmeans.labels_)
        df = pd.concat([df_symbol,cluster_labels],axis = 1)
        df.rename(columns = {0:"Third_Classification"},inplace = True)
        return df

    else:
        pass




def classification_statistics():
    conn_str_2 =pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=YOUR DIRECTORY\ClusteringDataBase.accdb;')
    df_risk_index = pd.read_sql("RiskIndex",acc_engine)
    df_performance_index = pd.read_sql("PerformanceIndex",acc_engine)
    df_fund_index = pd.read_sql("FundIndex",acc_engine)
    df_cluster_index = pd.read_sql("ClassifierIndex",acc_engine)
    df_cluster =Cluster_Classifier(4)
    df_merged = df_fund_index.merge(df_cluster, on = "Symbol")

    df_stats = df_merged.merge(df_risk_index,on = "ID")
    df_stats = df_stats.merge(df_performance_index, on = "ID")
    df_stats_group_cluster = df_stats.groupby("Third_Classification")


    # Cheack statistics for each variable
    skew_5_year = df_stats_group_cluster.Skew_5_year.agg(["count","max","min","median","mean"])
    std_5_year = df_stats_group_cluster.Std_5_year.agg(["count","max","min","median","mean"])
    max_dd_5_year = df_stats_group_cluster.Max_DD_5_year.agg(["count","max","min","median","mean"])
    cagr_5_year = df_stats_group_cluster.CAGR_5_year.agg(["count","max","min","median","mean"])
    tail_gain_5_year = df_stats_group_cluster.Expected_tail_gain_5_5_year.agg(["count","max","min","median","mean"])
    sharpe_ratio_5_year = df_stats_group_cluster.Sharpe_Ratio_5_year.agg(["count","max","min","median","mean"])
    starling_ratio_5_year = df_stats_group_cluster.Starling_5_year.agg(["count","max","min","median","mean"])


    # Updates the SQL DATABASE with the class for the funds
    #for i,x in zip(df_merged["ID"],df_merged["Third_Classification"]):
    #    update_third_class(i,x,conn_str_2)

    list_classes = [0,1,2,3,4]


    #df_merg = df.merge(df_risk_index,right_index = True, left_index = True)
    #df_merg = df_merg.merge(df_performance_index,right_index = True, left_index = True)

classification_statistics()
