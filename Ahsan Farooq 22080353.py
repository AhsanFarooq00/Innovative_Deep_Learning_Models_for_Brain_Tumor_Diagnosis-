import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score

os.environ["OMP_NUM_THREADS"] = '1'
pd.options.mode.chained_assignment = None

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame, skipping the first four rows and
    dropping certain columns.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pandas.DataFrame: Data loaded from the CSV.
    """
    print(file_path)
    data = pd.read_csv(file_path, skiprows=4)
    data = data.drop(columns=['Country Code','Indicator Name','Indicator Code','Unnamed: 67'])
    return data

def indicators_data(first_ind_name,Second_ind_name,df1,df2,Year):
    """
    

    Parameters
    ----------
    first_ind_name : String
        First Indicator Name.
    Second_ind_name : String
        Second Indicator Name.
    df1 : Pandas DataFrame
        First DataFrame.
    df2 : Pandas DataFrame 
        Second DataFrame.
    Year : String
        Data for Required Year.

    Returns
    -------
    df_cluster : Pandas DataFrame
        DataFrame For Clustering.

    """
    df1 = df1[['Country Name', Year]]
    df2 = df2[['Country Name',Year]]
    df = pd.merge(df1, df2,
                   on="Country Name", how="outer")
    df = df.dropna()
    df = df.rename(columns={Year+"_x": first_ind_name, Year+"_y": Second_ind_name})
    df_cluster = df[[first_ind_name, Second_ind_name]].copy()
    return df_cluster

def merge_indicators(indicator1, indicator2, data1, data2, year):
    """
    Merge two datasets based on the country name and specific year.
    
    Parameters:
        indicator1 (str): Name of the first indicator.
        indicator2 (str): Name of the second indicator.
        data1 (pandas.DataFrame): First dataset.
        data2 (pandas.DataFrame): Second dataset.
        year (str): Year of interest.
        
    Returns:
        pandas.DataFrame: Merged data with two indicators.
    """
    data1 = data1[['Country Name', year]]
    data2 = data2[['Country Name', year]]
    merged_data = pd.merge(data1, data2, on="Country Name", how="outer").dropna()
    merged_data = merged_data.rename(columns={year+"_x": indicator1, year+"_y": indicator2})
    return merged_data[['Country Name', indicator1, indicator2]]

def logistic_model(t, start, rate, midpoint):
    """
    Logistic growth model function.
    
    Parameters:
        t (array-like): Input times or years.
        start (float): The initial population or value.
        rate (float): The growth rate.
        midpoint (float): The midpoint of growth (inflection point).
        
    Returns:
        array-like: Modeled logistic growth values.
    """
    return start / (1.0 + np.exp(-rate * (t - midpoint)))

def fit_and_forecast(data, country, indicator, title, forecast_title, initial_params):
    """
    Fit data to a logistic model and forecast future values.
    
    Parameters:
        data (pandas.DataFrame): Data for fitting.
        country (str): Country name.
        indicator (str): Indicator name.
        title (str): Title for the fit plot.
        forecast_title (str): Title for the forecast plot.
        initial_params (tuple): Initial parameters for the logistic model fit.
        
    Returns:
        None: This function plots the results.
    """
    popt, _ = opt.curve_fit(logistic_model, data.index, data[country], p0=initial_params)
    data["Fitted Data"] = logistic_model(data.index, *popt)
    plt.figure()
    plt.plot(data.index, data[country], 'r-', label="Data")
    plt.plot(data.index, data["Fitted Data"], 'b--', label="Fit")
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title(title)
    plt.savefig(f'{country}_fit.png', dpi=300)
    
    future_years = np.linspace(1995, 2030)
    future_fit = logistic_model(future_years, *popt)
    error_bounds = err.error_prop(future_years, logistic_model, popt, _)
    plt.figure()
    plt.plot(data.index, data[country], 'r-', label="Data")
    plt.plot(future_years, future_fit, 'b--', label="Forecast")
    plt.fill_between(future_years, future_fit - error_bounds, future_fit + error_bounds, color="yellow", alpha=0.5)
    plt.legend(loc="upper left")
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title(forecast_title)
    plt.savefig(f'{country}_forecast.png', dpi=300)
    plt.show()

def extract_country_data(dataset, country, start, end):
    """
    Extract and filter data for a specific country and time range.
    
    Parameters:
        dataset (pandas.DataFrame): The dataset containing country data.
        country (str): Country name.
        start (int): Start year.
        end (int): End year.
        
    Returns:
        pandas.DataFrame: Filtered country data.
    """
    dataset = dataset.T
    dataset.columns = dataset.iloc[0]
    dataset = dataset.drop(['Country Name'])
    dataset = dataset[[country]]
    dataset.index = dataset.index.astype(int)
    dataset = dataset[(dataset.index > start) & (dataset.index <= end)]
    dataset[country] = dataset[country].astype(float)
    return dataset

def plot_clusters(data, x_label, y_label, graph_title, num_clusters, scaled_data, data_min, data_max):
    """
    Plot data clusters using KMeans clustering.
    
    Parameters:
        data (pandas.DataFrame): Original data.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        graph_title (str): Title for the graph.
        num_clusters (int): Number of clusters to form.
        scaled_data (pandas.DataFrame): Scaled data for clustering.
        data_min (float): Minimum value for scaling.
        data_max (float): Maximum value for scaling.
        
    Returns:
        np.ndarray: Cluster labels.
    """
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_
    centroids = ct.backscale(kmeans.cluster_centers_, data_min, data_max)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(data[x_label], data[y_label], c=labels, cmap="tab20")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="d", s=80)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(graph_title)
    plt.savefig('cluster_plot.png', dpi=300)
    plt.show()
    return labels

def evaluate_silhouette(data, max_clusters=10):
    """
    Evaluate and plot silhouette scores for different numbers of clusters.
    
    Parameters:
        data (pandas.DataFrame): Data for clustering.
        max_clusters (int): Maximum number of clusters to evaluate.
        
    Returns:
        None: This function plots the silhouette scores.
    """
    scores = []
    for clusters in range(2, max_clusters + 1):
        kmeans = cluster.KMeans(n_clusters=clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), scores, 'g-o')
    plt.title('Silhouette Scores for Varying Cluster Counts')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Loading the data
CO2_data = load_csv('CO2_emissions_metric_tons_per_capita.csv')
GDP_data = load_csv('GDP_per_capita_current_US$.csv')

# Merging the indicators into a single DataFrame for clustering
cluster_data = indicators_data('GDP per Capita (Current US$)', 'CO2 Emissions (Metric Tons per Capita)', GDP_data, CO2_data, '2020')

# Scaling data and evaluating silhouette scores
scaled_cluster_data, min_values, max_values = ct.scaler(cluster_data)
evaluate_silhouette(scaled_cluster_data, 12)

# Plotting clusters and updating the DataFrame with cluster labels
cluster_labels = plot_clusters(cluster_data, 'GDP per Capita (Current US$)', 'CO2 Emissions (Metric Tons per Capita)', 
                               'CO2 vs GDP per Capita in 2020', 3, scaled_cluster_data, min_values, max_values)

# Merging datasets for additional analysis
merged_data = merge_indicators('GDP per Capita (Current US$)', 'CO2 Emissions (Metric Tons per Capita)', GDP_data, CO2_data, '2020')
merged_data['Cluster Label'] = cluster_labels

# Filtering data for specific countries
specific_countries_data = merged_data[merged_data['Country Name'].isin(['India', 'China'])]
# Extracting, fitting, and predicting for China
China_data = extract_country_data(GDP_data, 'China', 1990, 2020)
China_data = China_data.fillna(0)
fit_and_forecast(China_data, 'China', 'GDP per Capita (Current US$)', 
                 "GDP per Capita in China 1990-2020", "GDP per Capita Forecast for China Until 2030", (1e5, 0.04, 1990))

# Extracting, fitting, and predicting for India
India_data = extract_country_data(GDP_data, 'India', 1990, 2020)
India_data = India_data.fillna(0)
fit_and_forecast(India_data, 'India', 'GDP per Capita (Current US$)', 
                 "GDP per Capita in India 1990-2020", "GDP per Capita Forecast for India Until 2030", (1e5, 0.04, 1990))