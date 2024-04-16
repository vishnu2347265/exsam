import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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



# Function for image processing tab
def image_processing():
    st.header('Image Processing')
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display original image
        original_image = Image.open(uploaded_image)
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Image processing options
        st.subheader("Image Processing Options")
        resize_option = st.checkbox("Resize")
        grayscale_option = st.checkbox("Grayscale")
        crop_option = st.checkbox("Crop")
        rotate_option = st.checkbox("Rotate")

        # Process the image based on selected options
        processed_image = original_image
        if resize_option:
            width = st.slider("Width", 10, 1000, 300)
            height = st.slider("Height", 10, 1000, 300)
            processed_image = processed_image.resize((width, height))

        if grayscale_option:
            processed_image = processed_image.convert("L")

        if crop_option:
            left = st.slider("Left", 0, processed_image.width, 0)
            top = st.slider("Top", 0, processed_image.height, 0)
            right = st.slider("Right", 0, processed_image.width, processed_image.width)
            bottom = st.slider("Bottom", 0, processed_image.height, processed_image.height)
            processed_image = processed_image.crop((left, top, right, bottom))

        if rotate_option:
            angle = st.slider("Angle", -180, 180, 0)
            processed_image = processed_image.rotate(angle)

        # Display processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)


# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if the value is a string
        # Tokenization
        tokens = nltk.word_tokenize(text.lower())
        
        # Removing special characters and keeping only alphanumeric tokens
        tokens = [token for token in tokens if token.isalnum()]
        
        # Removing stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        return " ".join(tokens)
    else:
        return ""  # Return an empty string if the value is not a string

# Preprocess the 'Review Text' column
df['preprocessed_text'] = df['Review Text'].apply(preprocess_text)


def text_similarity_analysis():
    st.header('Text Similarity Analysis')

    # Select division names for similarity analysis
    division_names = df['Division Name'].unique()
    selected_division = st.selectbox("Select Division Name", division_names)

    # Filter dataframe based on selected division
    division_df = df[df['Division Name'] == selected_division]

    # Select number of similar reviews to display
    num_similar_reviews = st.slider("Number of Similar Reviews", 1, 10, 5)

    # Select a review from the filtered dataframe
    selected_review = st.selectbox("Select a Review", division_df['Review Text'])

    # Preprocess the selected review
    preprocessed_selected_review = preprocess_text(selected_review)

    # Compute similarity with other reviews in the same division
    similarity_scores = []
    for review in division_df['preprocessed_text']:
        similarity_score = text_similarity(preprocessed_selected_review, review)
        similarity_scores.append(similarity_score)

    # Sort the reviews based on similarity scores
    similar_reviews_df = division_df.copy()
    similar_reviews_df['Similarity Score'] = similarity_scores
    similar_reviews_df = similar_reviews_df.sort_values(by='Similarity Score', ascending=False).head(num_similar_reviews)

    # Display similar reviews
    st.subheader("Similar Reviews:")
    for index, row in similar_reviews_df.iterrows():
        st.write(f"Rating: {row['Rating']}, Review Text: {row['Review Text']}")

# Function to compute text similarity using cosine similarity
def text_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix)[0, 1]


# Main function to run the app
def main():
    st.title("Women's Clothing E-Commerce")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ['3D Plot Visualization', 'Image Processing', 'Text Similarity Analysis'])

    if page == "3D Plot Visualization":
        display_demographic_data()
    elif page == "Image Processing":
        image_processing()

    elif page =='Text Similarity Analysis':
        text_similarity_analysis()

if __name__ == "__main__":
    main()
