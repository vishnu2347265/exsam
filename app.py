import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("WomensClothingE-CommerceReviews.csv")

df = load_data()

# Define functions to create visualizations

def display_demographic_data():
    st.header("Demographic Data Based on Age")
    fig = px.scatter_3d(df, x='Age', y='Rating', z='Positive Feedback Count', color='Rating', size='Positive Feedback Count',
                         hover_data=['Age', 'Rating', 'Positive Feedback Count'],
                         title='Age Distribution - Rating vs Positive Feedback Count')
    st.plotly_chart(fig)

def display_rating_distribution():
    st.header("Rating Distribution")
    rating_counts = df['Rating'].value_counts().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    fig = px.bar(rating_counts, x='Rating', y='Count', labels={'Rating': 'Rating', 'Count': 'Count'}, title='Distribution of Ratings')
    st.plotly_chart(fig)


def display_positive_feedback_distribution():
    st.header("Positive Feedback Count Distribution")
    fig = px.scatter(df, x='Positive Feedback Count', y='Age', title='Distribution of Positive Feedback Count')
    st.plotly_chart(fig)

# Main function to run the app
def main():
    st.title("Women's Clothing E-Commerce Dashboard")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Demographic Data", "Rating Distribution", "Recommended vs Not Recommended", "Positive Feedback Count"])

    if page == "Demographic Data":
        display_demographic_data()
    elif page == "Rating Distribution":
        display_rating_distribution()
    elif page == "Recommended vs Not Recommended":
        display_recommendation_comparison()
    elif page == "Positive Feedback Count":
        display_positive_feedback_distribution()

if __name__ == "__main__":
    main()
