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

# Functions
def header():
    st.set_page_config(page_title='FiscalNote StressLens', page_icon='https://fiscalnote.com/favicon.ico',
                       initial_sidebar_state="collapsed", layout="wide",
                       menu_items={'Get Help': 'https://fiscalnote.com/demo',
                                   'Report a bug': "https://fiscalnote.com/demo",
                                   'About': "https://fiscalnote.com/about"})
    st.markdown('<META NAME="robots" CONTENT="noindex,nofollow">', unsafe_allow_html=True)

    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True)

    st.markdown("""
            <style>
                   .block-container {
                        padding-top: 1rem;
                        padding-bottom: 0rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
            </style>
            """, unsafe_allow_html=True)
    # Set the CSS style for the purple rectangle and image overlay  position: relative; border-radius: 10px;

    css = """
        <style>
        .purple-rectangle {
            position: fixed; top: 0;
            background-color: #442547;
            padding: 50px 30px 5px;
            width: 100%;
            color: #ffffff;
            font-size: 24px;
            display: flex;
            align-items: center;
            z-index: 1000; /* Ensure the header is above other content */
        }
        .stress-text {
            color: #E0310B;
            font-family: "Uni Neue", sans-serif; 
        }
        .lens-text {
            color: white;
            font-family: "Uni Neue", sans-serif; /* Add Uni Neue font */
        }
        .image-overlay {
            width: 110px;
            margin-right: 2px;
        }
        </style>
    """
    # <img src="https://fiscalnote-marketing.s3.amazonaws.com/logo-FN-white-red.png" class="image-overlay">
    # Render the purple rectangle and image overlay using Markdown with HTML
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(
        f'''
        {css}
        <div class="purple-rectangle">
           <img src="https://fiscalnote-marketing.s3.amazonaws.com/logo-FN-white-red.png" class="image-overlay">
            <span class="stress-text">Stress</span><span class="lens-text">Lens</span>
        </div>
        ''',
        unsafe_allow_html=True
    )

# Define a function to format the numbers
def format_number(x):
    return f"{x:.0f}"  # Adjust formatting here

def create_complete_year_quarter_index(df: pd.DataFrame):
    min_year, max_year = int(df['calendar_year'].min()), int(df['calendar_year'].max())
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    year_quarters = [f'{year} {quarter}' for year in range(min_year, max_year + 1) for quarter in quarters]
    return year_quarters

def ensure_all_year_quarters(df: pd.DataFrame, year_quarters: List[str], group_by_cols: List[str]):
    all_combinations = pd.MultiIndex.from_product([df[col].unique() for col in group_by_cols] + [year_quarters], names=group_by_cols + ['calendar_year_quarter'])
    df = df.set_index(group_by_cols + ['calendar_year_quarter']).reindex(all_combinations).reset_index()
    return df
  
def plot_sector_helper(df: pd.DataFrame, group: str, property: str, prop_dict: Dict[str, str]):
    # Create a complete year/quarter index and ensure the DataFrame includes all year/quarters
    year_quarters = create_complete_year_quarter_index(df)
    df = ensure_all_year_quarters(df, year_quarters, [group])

    # Create an empty figure object
    fig = go.Figure()

    # Loop through each group (e.g., industry or sector) and add a trace for each
    for industry in df[group].unique():
        industry_data = df[df[group] == industry]
        
        # Add trace for each group with group name and the property in the legend
        fig.add_trace(go.Scatter(
            x=industry_data["calendar_year_quarter"],
            y=industry_data[property],  # Use the passed property argument
            mode='lines+markers',
            name=f"{industry} - {prop_dict.get(property)}"  # Include both sector and property in the legend
        ))

    # Update layout to make the plot wider and adjust titles and legends
    fig.update_layout(
        title=f"FS Stress {prop_dict.get(property)} Score by {group}",  # Dynamic title based on group and property
        xaxis_title="Calendar Year Quarter",
        yaxis_title=f"FS Stress {prop_dict.get(property)} Score",  # Dynamic y-axis label
        legend_title=group,  # Legend title will be the group name (e.g., 'Sector' or 'Industry')
        width=1200,  # Increase width of the plot
        height=600,  # Adjust height for better visualization
        showlegend=True  # Ensure the legend is visible
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=90)

    # Display plot in Streamlit
    st.plotly_chart(fig)

def plot_sector_topic_helper(df: pd.DataFrame, topics_of_interest: List[str], topic_colors: Dict[str, str], group: str, property: str, prop_dict: Dict[str, str]):
    # Create a complete year/quarter index and ensure the DataFrame includes all year/quarters
    year_quarters = create_complete_year_quarter_index(df)
    df = ensure_all_year_quarters(df, year_quarters, [group, 'topics'])

    # Get the unique sectors/groups to plot
    sectors = list(df[group].unique())

    # Function to plot a single figure
    def create_figure(sector: str):
        sector_data = df[df[group] == sector]
        fig = go.Figure()

        # Loop through each topic and add a trace for each topic in the given sector
        for topic in topics_of_interest:
            topic_data = sector_data[sector_data["topics"] == topic]
            color = topic_colors.get(topic, "black")  # Default to black if the topic color is not specified

            # Add trace with sector name in the legend to distinguish groups
            fig.add_trace(go.Scatter(
                x=topic_data["calendar_year_quarter"],
                y=topic_data[property],
                mode='lines+markers',
                name=f"{sector} - {topic} Score {prop_dict.get(property)}",  # Include sector name in the legend
                line=dict(color=color)
            ))

        # Update layout to adjust titles and legends
        fig.update_layout(
            title=f"{sector} FS Stress {prop_dict.get(property)} Score",
            width=1200,  # Set a standard width for the plot
            height=600,  # Set a standard height for the plot
            legend_title="Topics",  # Title for the legend
            showlegend=True,  # Ensure the legend is visible
            xaxis_title="Calendar Year/Quarter",
            yaxis_title=f"{property} Score"
        )

        # Update x-axis for better readability
        fig.update_xaxes(tickangle=90)

        return fig

    # Create and display a figure for each sector
    for sector in sectors:
        fig = create_figure(sector)
        st.plotly_chart(fig)


def plot_topics_helper(df: pd.DataFrame, topics_of_interest: List[str], topic_colors: Dict[str, str], property: str, prop_dict: Dict[str, str]):
    year_quarters = create_complete_year_quarter_index(df)
    df = ensure_all_year_quarters(df, year_quarters, ['topics'])

    fig = go.Figure()
    fig2 = go.Figure()

    for topic in topics_of_interest:
        topic_data = df[df["topics"] == topic]
        color = topic_colors.get(topic, "black")

        # Plotting the average FS Stress Score
        fig.add_trace(go.Scatter(
            x=topic_data["calendar_year_quarter"],
            y=topic_data[property],
            mode='lines+markers',
            name=f"{topic} Score {prop_dict.get(property)}",
            line=dict(color=color)
        ))

        # Adding a secondary y-axis for count
        fig2.add_trace(go.Scatter(
            x=topic_data["calendar_year_quarter"],
            y=topic_data["count"],
            mode='lines+markers',
            name=f"{topic} Count",
            line=dict(color=color, dash='dash')
        ))

    fig.update_layout(
        title=f"FS Stress {prop_dict.get(property)} Score by Topic",
        xaxis_title="Calendar Year Quarter",
        yaxis_title=f"FS Stress {prop_dict.get(property)} Score",
        legend_title="Topics",
        width=1200
    )
    fig.update_xaxes(tickangle=90)

    fig2.update_layout(
        title="Count by Topic",
        yaxis_title="Count",
        legend_title="Topics",
        width=1200
    )
    fig2.update_xaxes(tickangle=90)

    # Display both plots in Streamlit
    st.plotly_chart(fig)
    st.plotly_chart(fig2)


def plot_sector_topics_property(
    df: pd.DataFrame,
    topics_of_interest: Optional[List[str]],
    topic_colors: Optional[List[str]],
    group_sector: str,
    property: str,
):
    prop_dict = {
        "avg_fs_stress_score": "AVG",
        "std_fs_stress_score": "STD",
        "count": "count",
    }

    # Aggregating the data
    if group_sector and topics_of_interest:
        df_st = df.groupby(
            ["calendar_year", "calendar_quarter", "calendar_year_quarter", group_sector, "topics"], as_index=False
        ).agg(
            avg_fs_stress_score=("fs_stress_score", "mean"),
            std_fs_stress_score=("fs_stress_score", "std"),
            count=("fs_stress_score", "count"),
        )

        # Sort the dataframe by year and quarter
        df_st = df_st.sort_values(by=["calendar_year_quarter"]).reset_index(drop=True)

        plot_sector_topic_helper(
            df_st,
            topics_of_interest,
            topic_colors,
            group_sector,
            property,
            prop_dict,
        )

    elif group_sector:
        df_st = df.groupby(
            ["calendar_year", "calendar_quarter", "calendar_year_quarter", group_sector], as_index=False
        ).agg(
            avg_fs_stress_score=("fs_stress_score", "mean"),
            std_fs_stress_score=("fs_stress_score", "std"),
            count=("fs_stress_score", "count"),
        )
        # Sort the dataframe by year and quarter
        df_st = df_st.sort_values(by=["calendar_year_quarter"]).reset_index(drop=True)

        plot_sector_helper(
            df_st,
            group_sector,
            property,
            prop_dict,
        )

    else:
        df_st = df.groupby(["calendar_year", "calendar_quarter", "calendar_year_quarter", "topics"], as_index=False).agg(
            avg_fs_stress_score=("fs_stress_score", "mean"),
            std_fs_stress_score=("fs_stress_score", "std"),
            count=("fs_stress_score", "count"),
        )
        # Sort the dataframe by year and quarter
        df_st = df_st.sort_values(by=["calendar_year_quarter"]).reset_index(drop=True)

        plot_topics_helper(df_st, topics_of_interest, topic_colors, property, prop_dict)



def final_plot(df: pd.DataFrame, grouping: str, measure: str, topics_of_interest: List[str], topic_colors: Dict[str, str]):
    if grouping == "Topics":
        plot_sector_topics_property(df, topics_of_interest, topic_colors, None, measure)
    elif grouping == "Groups":
        plot_sector_topics_property(df, None, None, "group", measure)
    elif grouping == "Topics and Groups":
        plot_sector_topics_property(df, topics_of_interest, topic_colors, "group", measure)