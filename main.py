import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from keras.layers import Input, Embedding, Concatenate, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib as plt

# Load the dataset
data = pd.read_csv("a.csv")  # Replace with your dataset file path

# Preprocessing: Handling missing values
data.fillna(0, inplace=True)

user_id_to_index = {user_id: index for index, user_id in enumerate(data['user_id'].unique())}
product_id_to_index = {product_id: index for index, product_id in enumerate(data['product_id'].unique())}

user_indices = data['user_id'].map(user_id_to_index)
product_indices = data['product_id'].map(product_id_to_index)



# Assign weights to actions
weight_map = {'purchase': 2.0, 'like': 1.5, 'view': 1.0}
data['weighted_action'] = data['action'].map(weight_map)
ratings = data['weighted_action']

print(data[0:5])

# Create user-product interaction matrix with weighted actions
user_product_matrix = data.pivot_table(index='user_id', columns='product_id', values='weighted_action', fill_value=0)
print("User-Product Matrix Shape:", user_product_matrix.shape)

product_similarity_matrix= cosine_similarity(user_product_matrix)

# Compute the cosine similarity between the users based on their encoded features
user_similarity = 1 - pairwise_distances(user_product_matrix, metric='cosine')

def recommend_product_for_user(user_id, user_item_matrix, item_user_matrix, top_3):
    """Recommend top N product for a given user ID based on the collaborative filtering model"""
    # Get the row index for the user ID
    user_index = user_item_matrix.index.get_loc(user_id)
    # Get the user-item similarity scores for the user
    user_scores = user_item_matrix.iloc[user_index,:]
    # Sort the scores in descending order
    sorted_scores = user_scores.sort_values(ascending=False)
    # Get the top 3 product IDs
    top_product_id = sorted_scores.index[:top_3]
    # Convert the product IDs to integers
    top_product_id = [product_id for product_id in top_product_id]
    # Return the top product IDs
    return top_product_id

# Get a list of recommended product IDs for user with ID

given_user_id=input("enter the user_id:")
recommended_product_ids = recommend_product_for_user(given_user_id, user_product_matrix,product_similarity_matrix, top_n=3)
print(recommended_product_ids)


def get_product_titles(product_ids):
    # Load the dataset
    product_df = pd.read_csv('a.csv')

    # Filter the dataset to only include the desired product IDs
    recommended_product = product_df[product_df['product_id'].isin(product_ids)]

    # Extract the product titles from the filtered dataset
    product_titles = recommended_product[['title']].values.tolist()

    return product_titles


# Get the product titles for the recommended product IDs
recommended_product_titles = get_product_titles(product_ids=recommended_product_ids)
print(pd.DataFrame(recommended_product_titles[0:3]))





#graphical representation
import seaborn as sns


print("Weighted Cosine Similarity Matrix Shape:",product_similarity_matrix.shape)
print(product_similarity_matrix)

def plot_cosine_similarity_matrix(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.show()

# Compute and plot the weighted cosine similarity matrix
plot_cosine_similarity_matrix(product_similarity_matrix, "Weighted Cosine Similarity Matrix")

def generate_graphs():
    # Seller Ratings Distribution (Assuming 'seller_rating' is a column in the dataset)
    seller_ratings = data['seller_rating']
    plt.figure(figsize=(10, 6))
    plt.hist(seller_ratings, bins=30, color='green', alpha=0.7)
    plt.title('Seller Ratings Distribution')
    plt.xlabel('Seller Rating')
    plt.ylabel('Count')
    plt.show()

    # Count of Total Data
    total_data_count = len(data)
    print("Total Data Count:", total_data_count)

    # Unique Users and Products Count
    unique_users_count = len(user_id_to_index)
    unique_products_count = len(product_id_to_index)
    print("Unique Users Count:", unique_users_count)
    print("Unique Products Count:", unique_products_count)

    # Gender Distribution (Assuming 'gender' is a column in the dataset)
    gender_counts = data['gender'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Gender Distribution')
    plt.show()
    
    Age_distribution = data['age']
    plt.figure(figsize=(10, 6))
    plt.hist(Age_distribution, bins=30, color='green', alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('AGE GROUP')
    plt.ylabel('COUNT')
    plt.show()

# Call the function to generate graphs and display counts
generate_graphs()