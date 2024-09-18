import os
from datetime import datetime
from typing import *
import matplotlib.pyplot as plt
from matplotlib import colormaps
import plotly.graph_objects as go
import plotly.subplots as sp

import pandas as pd

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, explode, to_json, col
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType

import streamlit as st

from StressUtils import utils

utils.header()

# Set environment variables (replace with your values)
os.environ['DATABRICKS_HOST'] = st.secrets["databricks_credentials"]["databricks_host"]
os.environ['DATABRICKS_TOKEN'] = st.secrets["databricks_credentials"]["databricks_token"]
os.environ['DATABRICKS_CLUSTER_ID'] = st.secrets["databricks_credentials"]["databricks_cluster_id"]

# Initialize Spark session
spark = DatabricksSession.builder.profile("DEFAULT").getOrCreate()

# Define constants
SCHEMA_NAME = "hive_metastore.stresslens_data_labeled"
DELTA_TABLE_NAME = "TEXT_TOPICS_LABELED"
calendar_quarters = ["Q1", "Q2", "Q3", "Q4"]
tickers = sorted([row['ticker'] for row in spark.sql(f"SELECT DISTINCT ticker FROM {SCHEMA_NAME}.{DELTA_TABLE_NAME}").collect()])

# Streamlit UI elements for filtering
st.sidebar.title("Filters")

# Get number of groups
num_groups = st.sidebar.slider("Number of Groups", 1, 5, 1)
group_names = []
selected_tickers = []
ticker_group_name = {}
for i in range(1, num_groups + 1):
    group_name = st.sidebar.text_input(f"Group Name {i}", "")
    group_names.append(group_name)
    selected_tickers_group = st.sidebar.multiselect(f"Select Tickers for Group {i}", tickers)
    selected_tickers.extend(selected_tickers_group)
    for selected_ticker_group in selected_tickers_group:
        ticker_group_name[selected_ticker_group] = group_name

# Fetch unique values for dynamic filtering
current_year = int(datetime.now().year)
calendar_years = [i for i in range(2007, current_year+1)]
topics = ['Business_Model_Resilience', 'Director_Removal', 'Customer_Privacy', 'Employee_Engagement_Inclusion_And_Diversity', 'Systemic_Risk_Management', 'Business_Ethics', 'Product_Design_And_Lifecycle_Management', 'Data_Security', 'Physical_Impacts_Of_Climate_Change', 'Supply_Chain_Management', 'Selling_Practices_And_Product_Labeling', 'Human_Rights_And_Community_Relations', 'Customer_Welfare', 'Management_Of_Legal_And_Regulatory_Framework', 'Competitive_Behavior', 'Labor_Practices', 'Waste_And_Hazardous_Materials_Management', 'Critical_Incident_Risk_Management', 'Employee_Health_And_Safety', 'Water_And_Wastewater_Management', 'GHG_Emissions', 'Ecological_Impacts', 'Product_Quality_And_Safety', 'Air_Quality', 'Access_And_Affordability', 'Energy_Management', 'Company | Product News', 'Earnings', 'General News | Opinion', 'Personnel Change', 'Financials', 'Analyst Update', 'Politics', 'Legal | Regulation', 'Stock Commentary', 'IPO', 'M&A | Investments', 'Currencies', 'Fed | Central Banks', 'Treasuries | Corporate Debt', 'Gold | Metals | Materials', 'Energy | Oil', 'Macro', 'Stock Movement', 'Dividend', 'Markets']

# Update dynamic selections with fetched values
selected_topics = st.sidebar.multiselect("Select Topics", topics)
start_year = st.sidebar.selectbox("Start Year", calendar_years)
start_quarter = st.sidebar.selectbox("Start Quarter", calendar_quarters)
end_year = st.sidebar.selectbox("End Year", calendar_years)
end_quarter = st.sidebar.selectbox("End Quarter", calendar_quarters)
topics_threshold = st.sidebar.slider("Topics Threshold", 0.0, 1.0, 0.9)

measure = st.sidebar.selectbox("Measure", ["avg_fs_stress_score", "std_fs_stress_score", "count"], index=0)
grouping = st.sidebar.selectbox( "Grouping By", ["Topics", "Groups", "Topics and Groups"], index=0)

selected_topics_tuple = tuple(selected_topics)
selected_tickers_tuple = tuple(selected_tickers)

# Streamlit sidebar button
if st.sidebar.button("Submit"):
    column_names = ['id', 'sequence', 'ticker', 'label', 'date', 'quarter', 'year', 'speaker', 'title', 
    'text', 'fs_stress_score', ' calendar_year', 'calendar_quarter', 'calendar_year_quarter', 'created_at']
    select_columns = ", ".join(column_names)

    sql_query = f"""
                WITH filtered_data AS (
                    SELECT *
                    FROM {SCHEMA_NAME}.{DELTA_TABLE_NAME}
                    WHERE (ticker IN {selected_tickers_tuple}) 
                    AND (calendar_year_quarter >= '{start_year} {start_quarter}') 
                    AND (calendar_year_quarter <= '{end_year} {end_quarter}')
                ),
                exploded_topics AS (
                    SELECT *, EXPLODE(topics) AS topic
                    FROM filtered_data
                ),
                -- Remove duplicate texts and select certain topics with a certain threshold
                ranked_data AS (
                    SELECT {select_columns}, 
                           topic.label AS topics,
                           ROW_NUMBER() OVER (PARTITION BY id, speaker, text, fs_stress_score ORDER BY id, sequence) AS row_num
                    FROM exploded_topics
                    WHERE topic.score >= {topics_threshold}
                    AND topic.label IN {selected_topics_tuple}
                )
                SELECT {select_columns}, topics
                FROM ranked_data
                WHERE row_num = 1
                ORDER BY calendar_year, calendar_quarter, id, sequence
            """

    df = spark.sql(sql_query).toPandas()
    
    # Map tickers to group names
    # ticker_group_name = {ticker: group_names[i % num_groups] for i in range(num_groups+1) for ticker in selected_tickers}
    df['group'] = df['ticker'].map(ticker_group_name)


    # st.dataframe(df)
    # Apply the formatting
    st.dataframe(df.style.format({"year": utils.format_number, "sequence": utils.format_number}))

    # Extract unique topics and assign colors
    topics_of_interest = sorted(df['topics'].unique())
    colors = ["orange", "blue", "red", "green", "purple", "brown"]
    topic_colors = {topic: colors[ind % len(colors)] for ind, topic in enumerate(topics_of_interest)}

    utils.final_plot(df, grouping, measure, topics_of_interest, topic_colors)
