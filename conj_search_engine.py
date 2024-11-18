from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import string
import pandas as pd
import csv



def preprocess_text(text):
    """
    Preprocesses a given text by applying several text-cleaning steps, including case normalization, 
    stopword removal, stemming, and filtering unwanted tokens.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: A cleaned and preprocessed version of the input text.

    The function performs the following steps:
    1. Converts all characters to lowercase.
    2. Removes words that begin with the digit '0' using regular expressions.
    3. Tokenizes the text into individual words.
    4. Removes stopwords and punctuation, excludes non-alphanumeric tokens, filters out words shorter than 3 characters, and applies stemming to reduce words to their root form.
    5. Joins the cleaned words back into a single string, separated by spaces.

    Example:
        Input: "This is an example text with stopwords and punctuation, like 0hello!"
        Output: "exampl text stopword punctuat"
    """
    # Initialize stopwords, stemmer, and punctuation
    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove words that begin with the digit '0'
    text = re.sub(r'\b0\w*\b', '', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords, punctuation, apply stemming, and filter unwanted tokens
    processed_words = [
        stemmer.stem(word)  # Apply stemming
        for word in words
        if word not in stop_words and 
           word not in string.punctuation and 
           word.isalnum() and  # Exclude non-alphanumeric tokens
           len(word) > 2  # Exclude very short words
    ]
    
    # Return the processed words as a single string
    return ' '.join(processed_words)




def create_vocabulary(df):
    """
    Creates a vocabulary from the preprocessed restaurant descriptions in the DataFrame.
    
    The function concatenates all restaurant descriptions into a single string, 
    preprocesses the text (e.g., removes stopwords, punctuation, applies stemming), 
    extracts unique terms, assigns a term ID to each, and saves the vocabulary as a CSV file.
    
    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing a column 'description' with preprocessed restaurant descriptions.
        
    Returns:
    -------
    None
        This function saves the vocabulary as a CSV file but does not return any values.
    
    Example:
    --------
    create_vocabulary(df)
        This will create a vocabulary from the descriptions in the DataFrame and save it as 'vocabulary.csv'.
    """
    
    # Concatenate all preprocessed descriptions into a single string
    all_descriptions = ' '.join(df['description'].tolist())
    
    # Preprocess the concatenated text (e.g., remove stopwords, punctuation, apply stemming)
    processed_descriptions = preprocess_text(all_descriptions)
    
    # Get unique terms by splitting the processed text into words, removing duplicates, and sorting
    unique_terms = sorted(set(processed_descriptions.split()))
    
    # Create a DataFrame with 'term_id' and 'term' columns
    vocab_df = pd.DataFrame({
        'term_id': range(len(unique_terms)),  # Term ID for each unique term
        'term': unique_terms  # The unique terms themselves
    })
    
    # Save the vocabulary DataFrame to a CSV file
    vocab_df.to_csv('vocabulary.csv', index=False)
    
    # Print the number of unique terms created
    print(f"Created vocabulary with {len(vocab_df)} unique terms")




def build_inverted_index(processed_descriptions):
    """
    Builds an inverted index from the preprocessed restaurant descriptions.
    
    An inverted index maps each term (identified by a term ID) to a list of restaurant IDs 
    that contain the term in their description. This index helps quickly find which restaurants 
    are associated with a particular term. The function loads an existing vocabulary from a CSV file 
    and processes the descriptions to create the inverted index.

    Parameters:
    ----------
    processed_descriptions : list of str
        A list of preprocessed restaurant descriptions, where each description is a string of terms.

    Returns:
    -------
    dict
        An inverted index dictionary where keys are term IDs, and values are lists of restaurant IDs 
        that contain the respective term.
    
    Example:
    --------
    inverted_index = build_inverted_index(processed_descriptions)
        This will return an inverted index mapping term IDs to the list of restaurant IDs.
    """
    
    inverted_index = {}  # Initialize an empty dictionary for the inverted index
    
    # Load the existing vocabulary from the CSV file
    vocabulary = {}
    with open('vocabulary.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            vocabulary[row['term']] = int(row['term_id'])  # Map terms to their term IDs
    
    # Iterate through each restaurant description
    for restaurant_id, description in enumerate(processed_descriptions):
        # Split the description into terms (case-insensitive)
        for term in str(description).lower().split():
            if term in vocabulary:  # Only process terms that exist in the vocabulary
                term_id = vocabulary[term]  # Get the corresponding term ID
                
                # If this term ID is not in the inverted index, initialize it with an empty list
                if term_id not in inverted_index:
                    inverted_index[term_id] = []
                
                # If the restaurant is not already in the list for this term, add it
                if restaurant_id not in inverted_index[term_id]:
                    inverted_index[term_id].append(restaurant_id)
    
    return inverted_index



def search_restaurants(query, inverted_index, df):
    """
    Searches for restaurants that match the query terms based on their descriptions.
    
    The function processes the user's query, retrieves the relevant restaurant IDs from the 
    inverted index, and returns a DataFrame with the matching restaurants' details 
    (name, address, description, and website). Only the restaurants whose descriptions 
    contain all the query terms are considered as a match.

    Parameters:
    ----------
    query : str
        The search query entered by the user, which will be processed and matched against restaurant descriptions.
    
    inverted_index : dict
        The inverted index, where keys are term IDs and values are lists of restaurant IDs 
        that contain the respective terms in their descriptions.
    
    df : pandas.DataFrame
        The DataFrame containing restaurant information, including columns for 'restaurantName', 'address', 
        'description', and 'website'. This is used to retrieve the full details of matching restaurants.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing details of the restaurants that match the query terms. If no matches are found, 
        an empty DataFrame is returned.

    Example:
    --------
    result = search_restaurants("italian pizza", inverted_index, df)
        This will return a DataFrame with restaurants that have both "italian" and "pizza" in their descriptions.
    """
    
    # Load the vocabulary from the CSV file
    vocabulary = {}
    with open('vocabulary.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            vocabulary[row['term']] = int(row['term_id'])  # Map terms to term IDs
    
    # Preprocess the query and split into unique terms
    query_terms = set(preprocess_text(query).split())  # Preprocess the query and remove duplicates
    print(query_terms)
    
    # Find the intersection of restaurant IDs that contain all query terms
    matching_restaurant_ids = None
    for term in query_terms:
        if term in vocabulary:  # Check if term exists in vocabulary
            term_id = vocabulary[term]  # Get term ID
            if term_id in inverted_index:  # Check if term ID exists in the inverted index
                if matching_restaurant_ids is None:
                    matching_restaurant_ids = set(inverted_index[term_id])  # Initialize with the first term's matches
                else:
                    matching_restaurant_ids.intersection_update(set(inverted_index[term_id]))  # Keep only common matches
    print(matching_restaurant_ids)
    
    # If no matching restaurants, return an empty DataFrame
    if matching_restaurant_ids is None or not matching_restaurant_ids:
        return pd.DataFrame(columns=['Restaurant Name', 'Address', 'Description', 'Website'])
    
    # Construct the output DataFrame of matching restaurants
    matching_restaurants = pd.DataFrame({
        'Restaurant Name': df.loc[list(matching_restaurant_ids), 'restaurantName'],
        'Address': df.loc[list(matching_restaurant_ids), 'address'],
        'Description': df.loc[list(matching_restaurant_ids), 'description'],
        'Website': df.loc[list(matching_restaurant_ids), 'website']
    })
    
    return matching_restaurants



