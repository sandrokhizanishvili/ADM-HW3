from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import string
import pandas as pd
import numpy as np
from collections import Counter
from tqdm.notebook import tqdm
import pickle


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



def create_vocabulary(corpus, min_frequency=2, max_frequency=1800):
    """
    Creates a vocabulary from the given corpus by processing the text, filtering words
    based on their frequency, and saving the vocabulary to a CSV file.

    This function takes a collection of text descriptions (corpus), preprocesses the text,
    tokenizes it into words, counts the frequency of each word, and filters out words that 
    appear less frequently than `min_frequency` or more frequently than `max_frequency`.
    The resulting vocabulary (a list of unique, frequent words) is saved in a CSV file with 
    an associated term ID for each word.

    Parameters:
    ----------
    corpus : list of str
        A list of preprocessed text descriptions (e.g., restaurant descriptions) from which 
        the vocabulary will be created.
    
    min_frequency : int, optional, default=2
        The minimum frequency a word must appear to be included in the vocabulary. Words 
        appearing fewer times than this value will be filtered out.
    
    max_frequency : int, optional, default=1800
        The maximum frequency a word can have to be included in the vocabulary. Words 
        appearing more times than this value will be filtered out.

    Returns:
    -------
    None
        The function does not return any value but saves the vocabulary to a CSV file (`vocabulary.csv`).
    
    Example:
    --------
    corpus = ["modern italian cuisine", "seasonal dishes", "best italian pizza"]
    create_vocabulary(corpus, min_frequency=2, max_frequency=1800)
    """
    
    # Concatenate all preprocessed descriptions into a single list
    all_descriptions = ' '.join(corpus)
    
    # Preprocess the concatenated text
    processed_descriptions = preprocess_text(all_descriptions)
    
    # Tokenize the preprocessed descriptions and calculate word frequencies
    word_list = processed_descriptions.split()
    word_counts = Counter(word_list)  # Count the frequency of each word
    
    # Filter out words with frequency below the specified threshold
    frequent_words = [word for word, count in word_counts.items() if count >= min_frequency and count <= max_frequency]
    
    # Create vocabulary DataFrame with only frequent words
    unique_terms = sorted(set(frequent_words))  # Get unique terms and sort them
    vocab_df = pd.DataFrame({
        'term_id': range(len(unique_terms)),  # Assign a unique term ID to each term
        'term': unique_terms  # List of terms
    })
    
    # Save vocabulary to CSV
    vocab_df.to_csv('vocabulary_rse.csv', index=False)
    
    # Print the size of the created vocabulary
    print(f"Created vocabulary with {len(vocab_df)} unique terms (frequency >= {min_frequency}) and (frequency <= {max_frequency})")



def build_inverted_index(tfidf_matrix, terms,  file_name='inverted_index'):
    """
    Creates an inverted index from the given TF-IDF matrix and list of terms, 
    and saves it to a file (pickle format).

    An inverted index maps each term to a list of document IDs and their corresponding
    TF-IDF scores in that document. This function iterates through each term, extracts the 
    non-zero TF-IDF scores from the matrix, and stores the document IDs and TF-IDF scores.
    
    Args:
    - tfidf_matrix : The sparse matrix representing the TF-IDF values 
      of the documents (rows) and terms (columns).
    - terms (list of str): A list of terms corresponding to the columns of the TF-IDF matrix.
    - file_name (str): The name of the file to save the inverted index (without extension).
    
    Returns:
    - dict: An inverted index where each term maps to a list of tuples, where each tuple contains 
      a document ID and the corresponding TF-IDF score.
    """
    
    # Initialize an empty dictionary to hold the inverted index
    inverted_index = {}
    
    # Iterate over all terms and their respective indices in the 'terms' list
    for term_idx, term in tqdm(list(enumerate(terms)), desc="Building Inverted Index"):
        
        # Extract the column for the current term from the TF-IDF matrix
        term_column = tfidf_matrix[:, term_idx]
        
        # Find the non-zero entries (i.e., documents where the term appears)
        non_zero_entries = term_column.nonzero()[0]
        
        # For each non-zero entry, store the document ID and corresponding TF-IDF score
        inverted_index[term] = [
            (doc_id, term_column[doc_id, 0]) for doc_id in non_zero_entries
        ]
        

    # Save as a pickle file
    with open(f"{file_name}.pkl", 'wb') as f:
        pickle.dump(inverted_index, f)
    print(f"Inverted index saved as {file_name}.pkl")
    
    return inverted_index


def rank_restaurants_by_query_similarity(tfidf_matrix, tfidf_query,  terms, inverted_index, df, k=5):
    """
    Ranks restaurants based on a query by calculating cosine similarity between the query and documents
    using the inverted index and TF-IDF matrix.

    The function calculates the cosine similarity between the query and the documents using the inverted index, and returns the top-k matching restaurants
    along with their similarity scores.

    Args:
    - tfidf_matrix : The sparse matrix representing the TF-IDF values of the documents.
    - terms (list of str): A list of terms corresponding to the columns of the TF-IDF matrix.
    - inverted_index (dict): A dictionary mapping terms to lists of tuples (doc_id, tfidf_score).
    - df (pandas.DataFrame): The DataFrame containing restaurant data, including 'restaurantName', 'address',
                              'description', and 'website'.
    - k (int): The number of top matching documents (restaurants) to return (default is 5).
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the top-k matching restaurants with their 'Restaurant Name',
                         'Address', 'Description', 'Website', and 'Similarity Score'.
    """
    
    
    # Calculate cosine similarity using inverted index
    doc_scores = {}
    query_norm = np.linalg.norm(tfidf_query)

    for term_idx, query_score in enumerate(tfidf_query):
        if query_score > 0:
            term = terms[term_idx]
            if term in inverted_index:
                for doc_id, tfidf_doc_score in inverted_index[term]:
                    # Accumulate dot product scores for each document
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += query_score * tfidf_doc_score
    
    # Normalize by document norms to calculate cosine similarity
    cosine_similarities = {}
    for doc_id, dot_product in doc_scores.items():
        doc_norm = np.linalg.norm(tfidf_matrix[doc_id].toarray().flatten())
        cosine_similarities[doc_id] = dot_product / (query_norm * doc_norm)

    # Sort and get top-k results
    top_k_docs = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)[:k]

    # Prepare the DataFrame with top-k results
    top_k_indices = [i[0] for i in top_k_docs]  # Top k indices
    top_k_scores = [i[1] for i in top_k_docs]  # Top k similarity scores

    top_k_df = df.loc[top_k_indices, ['restaurantName', 'address', 'description', 'website']] \
        .rename(columns={'restaurantName': 'Restaurant Name', 'address': 'Address', 'description': 'Description', 'website': 'Website'})
    top_k_df['Similarity Score'] = top_k_scores

    return top_k_df


