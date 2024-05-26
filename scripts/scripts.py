import fasttext
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
from gensim.models.fasttext import load_facebook_model
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from difflib import SequenceMatcher
import shutil
import logging

################################################################
#### logging
################################################################
current_directory = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(current_directory, 'scripts.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
def log(message, level='info'):
    if level == 'debug':
        logger.debug(message)
    elif level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    elif level == 'critical':
        logger.critical(message)
    else:
        logger.info(message)
logger = logging.getLogger()

################################################################
#### static variables
################################################################
languages = {
    'en': {
        'full_name': 'English', 
        'model_path': '../models/cc.en.300.bin',
        'exclude_words': ["man", "woman", "Phil", "marv", "ole", "owld", "utd"],
        # "particuler", "legendry" -- archaic spellings
        'genders': ['man', 'woman'],
        'determiners': ['the'],
        'personhood_word':'person',
        'depersonalized_genders':['masculinity', 'femininity'],
    },
    'es': {
        'full_name': 'Spanish', 
        'model_path': '../models/cc.es.300.bin',
        'exclude_words': ["hombre", "mujer"],
        'genders': ['hombre', 'mujer'],
        'determiners': ['el', 'la'],
        'personhood_word':'persona',
        'depersonalized_genders':['masculinidad', 'femininidad'],
    },
    'de': {
        'full_name': 'German', 
        'model_path': '../models/cc.de.300.bin',
        'exclude_words': ["Mann", "Frau", "mfG", "ein"],
        'genders': ['Mann', 'Frau'],
        'determiners': ['der', 'die', 'das'],
        'personhood_word':'Individuum',
        'depersonalized_genders':['Männlichkeit', 'Weiblichkeit'],
    }
}

columns = {
    'masculine':'masculine_score',
    'feminine':'feminine_score',
}

# Auto-generating parquet_paths based on languages
parquet_paths = {lang: f"../materials/adjectives/{lang}_adjectives.parquet" for lang in languages.keys()}

targets = ['masculine', 'feminine']

nouns_df = pd.read_csv('../materials/nouns.csv')

################################################################
#### loading models
################################################################
def load_model(language, method="normal"):
    """Loads the specified language model into memory.

    This function loads a model based on the language and method provided.
    The method can be either 'normal' or 'facebook', which affects how the model is loaded.

    Args:
        language (str): The two-letter code of the language for which to load the model.
        method (str, optional): The method of loading the model. Defaults to 'normal'. Options are:
            - 'normal': Load a model given a filepath and return a model object.
            - 'facebook': Load the model from Facebook's native fasttext .bin output file.

    Returns:
        The loaded model.

    Raises:
        KeyError: If the language code is not found in the languages dictionary.
        FileNotFoundError: If the model file does not exist at the specified path.

    logs:
        The path of the loaded model and a confirmation message.
    """
    log(f'Loading model for {language} with method: {method}')
    model_path = languages[language]['model_path']
    log(model_path)
    if method == 'normal': 
        model = fasttext.load_model(model_path)
    else:
        model = load_facebook_model(model_path)
    log(f'Finished loading model for {language} with method: {method}')
    return model

#### WARNING: this line loads large embedding models into your memory.
models = {lang: load_model(lang, method='normal') for lang in languages.keys()}

################################################################
#### basic functions
################################################################
def cossim(vec1, vec2):
    """Calculates the cosine similarity between two vectors.

    This function computes the cosine similarity between two vectors `vec1` and `vec2`.
    Cosine similarity is a measure of similarity between two non-zero vectors of an inner product
    space that measures the cosine of the angle between them.

    Args:
        vec1 (list of float): First vector.
        vec2 (list of float): Second vector.

    Returns:
        float: Cosine similarity between `vec1` and `vec2`, ranging from -1 meaning exactly opposite,
               to 1 meaning exactly the same, with 0 indicating orthogonality (decorrelation), and
               in-between values indicating intermediate similarity or dissimilarity.

    """
    dot_product = sum(a*b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum([val**2 for val in vec1]))
    magnitude2 = math.sqrt(sum([val**2 for val in vec2]))
    return dot_product / (magnitude1 * magnitude2)

def get(model, word):
    """Retrieves the word embedding for a given word from a model.

    This function returns the word embedding for `word` from the specified `model`.
    Word embeddings are a type of word representation that allows words to be represented
    as vectors in a continuous vector space.

    Args:
        model (Model): The model from which to retrieve the word embedding.
        word (str): The word for which to retrieve the embedding.

    Returns:
        ndarray: The embedding of `word` as a dense vector.

    Raises:
        KeyError: If `word` is not in the model's vocabulary.

    """
    return model.get_word_vector(word)

def load_dataframe(file_path):
    """Loads a DataFrame from a file at `file_path`.

    This function supports loading from CSV and Parquet files. It checks the file extension
    and loads the DataFrame accordingly. 

    Args:
        file_path (str): The path to the file to load.

    Returns:
        DataFrame: A pandas DataFrame containing the data from `file_path`.

    Raises:
        ValueError: If the file format is not supported.

    """
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format")

################################################################
#### Wiktionary crawling
################################################################
def fetch_html_content(url):
    """Fetches the HTML content from a given URL.

    This function sends a GET request to the specified `url` and returns its HTML content
    as a string. It raises an HTTPError if the request encounters an error.

    Args:
        url (str): The URL from which to fetch the HTML content.

    Returns:
        str: The HTML content of the webpage.

    Raises:
        HTTPError: If the request fails due to client or server HTTP errors.

    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_adjectives(soup):
    """Extracts adjectives from a BeautifulSoup object containing Wiktionary page content.

    This function parses a BeautifulSoup object for adjectives listed on a Wiktionary page. It filters out
    entries containing digits, spaces, hyphens, plus signs, ampersands, apostrophes, periods, and parentheses.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the Wiktionary page.

    Returns:
        list of str: A list of adjectives extracted from the page.
    """
    mw_category_groups = soup.find_all(class_="mw-category-group")
    adjectives = []
    for group in mw_category_groups:
        li_tags = group.find_all('li')
        for li in li_tags:
            adjective = li.get_text()
            if not any(char.isdigit() for char in adjective) and " " not in adjective and "-" not in adjective and "+" not in adjective and "&" not in adjective and "'" not in adjective and "." not in adjective and "(" not in adjective:
                adjectives.append(adjective)
    return adjectives

def find_next_page_url(soup):
    """Finds and returns the URL of the next page in a Wiktionary category listing.

    This function searches for a 'next page' link in a BeautifulSoup object and constructs the full URL
    to the next page of a Wiktionary category listing if such a link exists.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the current page.

    Returns:
        str or None: The URL of the next page if found, otherwise None.
    """
    next_page_link = soup.find("a", string="next page")
    return 'https://en.wiktionary.org' + next_page_link.get('href') if next_page_link else None

def extract_adjectives(language, url, max_pages=None):
    """Extracts adjectives for a given language from Wiktionary starting from a specific URL.

    This function crawls Wiktionary pages to collect adjectives for the specified language. It iterates through pages,
    parsing and collecting adjectives until a maximum number of pages is reached or no further pages are found.

    Args:
        language (str): The language for which to extract adjectives.
        url (str): The starting URL for crawling.
        max_pages (int, optional): The maximum number of pages to crawl. If None, crawls without limit.

    Returns:
        list of str: A list of collected adjectives.
    """
    all_adjectives = []
    page_count = 0
    while url and (max_pages is None or page_count < max_pages):
        html_content = fetch_html_content(url)
        soup = BeautifulSoup(html_content, 'html.parser')
        adjectives = parse_adjectives(soup)
        all_adjectives.extend(adjectives)
        url = find_next_page_url(soup)
        page_count += 1
    return all_adjectives

def save_adjectives_to_parquet(adjectives, language_code, file_path):
    """Saves a list of adjectives to a Parquet file.

    Takes a list of adjectives and their corresponding language code, then saves this information into a Parquet file. 
    Parquet is chosen for its efficiency in both storage and speed when handling data operations within pandas DataFrames.

    Args:
        adjectives (list of str): The list of adjectives to be saved.
        language_code (str): The ISO 639-1 language code representing the language of the adjectives.
        file_path (str): The path to where the Parquet file will be saved, including the file name and its extension.

    Raises:
        FileNotFoundError: If the specified file_path directory does not exist.
        ValueError: If `adjectives` or `language_code` are empty.
    """
    df = pd.DataFrame(adjectives, columns=['Adjective'])
    df['Language'] = language_code
    df.to_parquet(file_path, index=False)

def find_adjective_definition(adjective):
    """Finds and returns the definition of an adjective using Wiktionary.

    This function searches Wiktionary for the given adjective and extracts the first definition it finds. It's designed to 
    handle English words but can be adjusted for other languages by modifying the URL accordingly.

    Args:
        adjective (str): The adjective for which to find the definition.

    Returns:
        str: The first definition of the adjective if found; otherwise, a message indicating the definition was not found.
    """
    url = f"https://en.wiktionary.org/wiki/{adjective}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        definition_section = soup.find('span', {'id': 'Adjective'})
        if definition_section:
            definition_list = definition_section.find_next('ol')
            if definition_list:
                first_item = definition_list.find('li')
                if first_item:
                    definition = first_item.get_text(separator=' ', strip=True).split('.')[0]
                    return definition
    except requests.HTTPError as e:
        return f"Error retrieving page for {adjective}: {e}"
    except Exception as e:
        return f"Error processing {adjective}: {e}"

    return "Definition not found."

################################################################
#### similarity calculations
################################################################
def calculate_adjective_similarities(language_code, method=cossim):
    """Calculates and adds gender-related similarity scores to adjectives for a given language.

    This function calculates gender-related similarity scores for each adjective in a specified language using a word embedding model. It loads adjectives from a Parquet file, computes similarities with predefined gender-related target words using a specified similarity function (defaulting to cosine similarity), and updates the DataFrame with these similarity scores for various categories, including masculine, feminine, exclusive masculine, exclusive feminine, depersonalized masculine, and depersonalized feminine similarities. Finally, it saves the updated DataFrame back to a Parquet file.

    Args:
        language_code (str): The ISO 639-1 language code for the target language.
        method (function, optional): The function to use for calculating similarity between vectors. Defaults to cosine similarity.

    Note:
        This function requires a predefined mapping in `parquet_paths` for loading the DataFrame, a `models` dictionary with loaded word embedding models for each language, and `languages` dictionary containing target words for each gender category in the specified language.
    """
    # Load the Parquet file into a DataFrame
    parquet_file_path = parquet_paths[language_code]
    df = pd.read_parquet(parquet_file_path)

    # Load the word embedding model
    language_data = languages[language_code]
    model = models[language_code]

    # Initialize columns for similarities
    df['masculine_similarity'] = 0.0
    df['feminine_similarity'] = 0.0
    df['exclusive_masculine_similarity'] = 0.0
    df['exclusive_feminine_similarity'] = 0.0
    df['depersonalized_masculine_similarity'] = 0.0
    df['depersonalized_feminine_similarity'] = 0.0

    # Get target word embeddings
    masculine_target = get(model, language_data['genders'][0])
    feminine_target = get(model, language_data['genders'][1])
    neuter_target = get(model, language_data['personhood_word'])

    # Calculate cosine similarities
    for index, row in df.iterrows():
        word_vec = get(model, row['Adjective'])

        # Regular similarities
        df.at[index, 'masculine_similarity'] = method(word_vec, masculine_target)
        df.at[index, 'feminine_similarity'] = method(word_vec, feminine_target)
        df.at[index, 'neuter_similarity'] = method(word_vec, neuter_target)
        df.at[index, 'depersonalized_masculine_similarity'] = method(word_vec, get(models[language_code], language_data['depersonalized_genders'][1]))
        df.at[index, 'depersonalized_feminine_similarity'] = method(word_vec, get(models[language_code], language_data['depersonalized_genders'][0]))

    # Save the updated DataFrame back to Parquet
    df.to_parquet(parquet_file_path)

def calculate_similarity(model, words, target_words, ref_group_label, language, ref_association, target_group):
    """Calculates the cosine similarity between sets of word vectors and target vectors.

    This function computes the cosine similarity for each pair consisting of a 'reference word' from the given 'words' 
    list and a 'target word' from the 'target_words' list. It uses a specified model to get the vector representations 
    of these words. The results, including language, reference group label, reference association, reference word, target 
    group, target word, and the calculated cosine similarity, are compiled into a DataFrame.

    Args:
        model: The word embedding model used to get vector representations.
        words (list of str): The list of reference words to compare.
        target_words (list of str): The list of target words to compare against the reference words.
        ref_group_label (str): Label for the reference group (e.g., 'Adjectives').
        language (str): The language of the words being compared.
        ref_association (str): Association of the reference group (e.g., 'Positive', 'Negative').
        target_group (str): Label for the target word group (e.g., 'Gender').

    Returns:
        DataFrame: A pandas DataFrame containing the language, reference group label, reference association, reference 
        word, target group, target word, and the cosine similarity for each comparison.
    """
    results = []
    for word in words:
        word_vec = get(model, word)
        for target_word in target_words:
            target_vec = get(model, target_word)
            similarity = cossim(word_vec, target_vec)
            results.append({
                'LANGUAGE': language,
                'REFERENCE GROUP': ref_group_label,
                'REFERENCE ASSOCIATION': ref_association,
                'REFERENCE WORD': word,
                'TARGET GROUP': target_group,
                'TARGET WORD': target_word,
                'COSINE SIMILARITY': similarity
            })
    return pd.DataFrame(results)

################################################################
#### data wrangling
################################################################
def select_top_words(language_code, method, num_rows=1000, semantic_differential_vectors='gender1-gender2'):
    """Selects top adjectives based on gender-related scores from a dataset for a given language.

    This function loads adjectives from a Parquet file and applies either a semantic differential method or 
    cosine similarity to score them based on gender-related dimensions. It then selects the top scoring adjectives 
    for masculine and feminine categories while excluding specific words. The selected adjectives are saved to 
    separate Parquet files for each gender category.

    Args:
        language_code (str): The ISO 639-1 language code for the target language.
        method (str): The method to apply for scoring adjectives. Can be 'semantic_differential' or 'cosine_similarity'.
        num_rows (int, optional): The number of top adjectives to select for each gender category. Defaults to 1000.
        semantic_differential_vectors (str, optional): Specifies the vectors to use for the semantic differential 
            method. Defaults to 'gender1-gender2'.

    Returns:
        tuple: Two pandas DataFrames, the first containing the selected masculine adjectives, and the second containing 
        the selected feminine adjectives.
    """
    # Load the Parquet file into a DataFrame
    parquet_file_path = parquet_paths[language_code]
    df = pd.read_parquet(parquet_file_path)

    # Access language-specific data
    language_data = languages[language_code]

    # Implement the semantic differential method
    if method == 'semantic_differential':
        if semantic_differential_vectors == 'gender1-gender2':
            df['masculine_score'] = df['masculine_similarity'] - df['feminine_similarity']
            df['feminine_score'] = df['feminine_similarity'] - df['masculine_similarity']
        elif semantic_differential_vectors == 'gender-person':
            df['masculine_score'] = df['masculine_similarity'] - df['neuter_similarity']
            df['feminine_score'] = df['feminine_similarity'] - df['neuter_similarity']
        elif semantic_differential_vectors == 'gender-Gender':
            df['masculine_score'] = df['masculine_similarity'] - df['depersonalized_feminine_similarity']
            df['feminine_score'] = df['feminine_similarity'] - df['depersonalized_masculine_similarity']
    elif method == 'cosine_similarity':
        df['masculine_score'] = df['masculine_similarity']
        df['feminine_score'] = df['feminine_similarity']

    # Filter out excluded words
    exclude_list = language_data['exclude_words']
    df = df[~df['Adjective'].isin(exclude_list)]

    # Initialize empty DataFrames for selected words
    selected_masculine = pd.DataFrame(columns=df.columns)
    selected_feminine = pd.DataFrame(columns=df.columns)

    # Select top adjectives for each gender
    while len(selected_masculine) < num_rows or len(selected_feminine) < num_rows:
        if len(selected_masculine) < num_rows:
            top_masculine = df.sort_values(by='masculine_score', ascending=False).head(num_rows * 2)
            top_masculine = top_masculine[~top_masculine['Adjective'].isin(selected_feminine['Adjective'])].head(num_rows - len(selected_masculine))
            selected_masculine = pd.concat([selected_masculine, top_masculine])

        if len(selected_feminine) < num_rows:
            top_feminine = df.sort_values(by='feminine_score', ascending=False).head(num_rows * 2)
            top_feminine = top_feminine[~top_feminine['Adjective'].isin(selected_masculine['Adjective'])].head(num_rows - len(selected_feminine))
            selected_feminine = pd.concat([selected_feminine, top_feminine])

    # Save the selected words to Parquet files
    masculine_file_path = f'../materials/adjectives/{language_code}_masculine_adjectives.parquet'
    feminine_file_path = f'../materials/adjectives/{language_code}_feminine_adjectives.parquet'
    selected_masculine.to_parquet(masculine_file_path, index=False)
    selected_feminine.to_parquet(feminine_file_path, index=False)

    return selected_masculine, selected_feminine

def duplicate_spanish_adjectives(df, association):
    """Generates alternate forms of Spanish adjectives based on gender and updates the DataFrame.

    This function adds an 'Alternate Form' column to a DataFrame containing Spanish adjectives. It generates the 
    opposite gender form of each adjective based on its ending ('o' for masculine to 'a' for feminine and vice versa) 
    and updates the DataFrame with these alternate forms. Finally, it saves the updated DataFrame to a Parquet file, 
    differentiating the file name based on the association (masculine or feminine) of the input adjectives.

    Args:
        df (DataFrame): The DataFrame containing Spanish adjectives.
        association (str): The gender association ('masculine' or 'feminine') of the input DataFrame.

    Returns:
        DataFrame: The updated DataFrame including the 'Alternate Form' column with opposite gender endings.

    Raises:
        ValueError: If the association parameter is not 'masculine' or 'feminine'.
    """
    # Add 'Alternate Form' column based on gender association
    df['Alternate Form'] = df['Adjective'].apply(lambda x: x[:-1] + 'a' if x.endswith('o') else x[:-1] + 'o')

    # Save the updated DataFrame to a Parquet file
    file_path = f'../materials/adjectives/es_{association}_adjectives.parquet'
    df.to_parquet(file_path, index=False)

    return df

def create_control_test_dataframe(lang_code, nouns_df, model):
    """Creates a control test DataFrame for a specified language.

    This function generates a DataFrame for testing control groups in linguistic research, specifically focusing on nouns 
    and adjectives. It computes the cosine similarity of nouns and adjectives in the specified language to target words 
    defined in a global language configuration, accounting for grammatical gender and other associations.

    Args:
        lang_code (str): The ISO code for the target language.
        nouns_df (pd.DataFrame): A DataFrame containing nouns and their associations/grammatical genders.
        model: The word embedding model used to retrieve vector representations.

    Returns:
        pd.DataFrame: A combined DataFrame with cosine similarity scores for nouns and adjectives across specified 
        target groups ('genders', 'determiners').
    """
    control_test_data = []

    # Process nouns
    nouns_lang_df = nouns_df[nouns_df['LANGUAGE'] == lang_code]
    for noun_gender in nouns_lang_df['ASSOCIATION/GRAMMATICAL GENDER'].unique():
        nouns = nouns_lang_df[nouns_lang_df['ASSOCIATION/GRAMMATICAL GENDER'] == noun_gender]['WORD'].tolist()
        for target_group in ['genders', 'determiners']:
            target_words = languages[lang_code][target_group]
            for target_word in target_words:
                data = calculate_similarity(model, nouns, [target_word], 'nouns', lang_code, noun_gender, target_group)
                control_test_data.append(data)

    # Process adjectives for each gender association
    for adj_association in ['masculine', 'feminine']:  # Assuming 'targets' is meant to be these associations
        adjectives_df = pd.read_parquet(f'../materials/adjectives/{lang_code}_{adj_association}_adjectives.parquet')
        adjectives = adjectives_df['Adjective'].tolist()
        for target_group in ['genders', 'determiners']:
            target_words = languages[lang_code][target_group]
            for target_word in target_words:
                data = calculate_similarity(model, adjectives, [target_word], 'adjectives', lang_code, adj_association, target_group)
                control_test_data.append(data)

            if lang_code == 'es' and 'Alternate Form' in adjectives_df.columns:
                alternate_forms = adjectives_df['Alternate Form'].dropna().tolist()
                for alternate_form in alternate_forms:
                    data = calculate_similarity(model, [alternate_form], [target_word], 'adjectives', lang_code, adj_association, target_group)
                    control_test_data.append(data)

    combined_data = pd.concat(control_test_data, ignore_index=True)
    return combined_data

def create_experimental_test_dataframe(lang_code, nouns_df, model, use_groupby=False):
    """Creates an experimental test DataFrame for a specified language.

    This function generates a DataFrame for conducting experimental linguistic tests, focusing on the relationship 
    between nouns and adjectives, including alternate forms of adjectives in languages like Spanish. It calculates 
    cosine similarities between nouns and adjectives, and optionally groups results to average similarities across 
    grammatical genders and adjective associations.

    Args:
        lang_code (str): The ISO code for the target language.
        nouns_df (pd.DataFrame): A DataFrame containing nouns and their grammatical genders.
        model: The word embedding model used to retrieve vector representations.
        use_groupby (bool, optional): A flag to determine if the output should be grouped by grammatical gender 
            of nouns and gender association of adjectives. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with cosine similarity scores, including additional metadata such as the language, 
        grammatical gender of nouns, and gender association of adjectives.
    """
    experimental_test_data = []

    nouns_lang_df = nouns_df[nouns_df['LANGUAGE'] == lang_code]

    for adj_association in ['masculine', 'feminine']:
        adjectives_df = pd.read_parquet(f'../materials/adjectives/{lang_code}_{adj_association}_adjectives.parquet')
        adjectives = adjectives_df['Adjective'].tolist()

        if lang_code == 'es':
            alternate_forms = adjectives_df['Alternate Form'].dropna().tolist()

            for noun_gender in nouns_lang_df['ASSOCIATION/GRAMMATICAL GENDER'].unique():
                nouns = nouns_lang_df[nouns_lang_df['ASSOCIATION/GRAMMATICAL GENDER'] == noun_gender]['WORD'].tolist()
                for noun in nouns:
                    for alternate_form in alternate_forms:
                        similarity = cossim(model.get_word_vector(noun), model.get_word_vector(alternate_form))
                        experimental_test_data.append({
                            'LANGUAGE': lang_code,
                            'GRAMMATICAL GENDER OF NOUN': noun_gender,
                            'NOUN': noun,
                            'ADJECTIVE': alternate_form,
                            'COSINE SIMILARITY': similarity,
                            'GENDER ASSOCIATION OF ADJECTIVE': adj_association
                        })

        for noun_gender in nouns_lang_df['ASSOCIATION/GRAMMATICAL GENDER'].unique():
            nouns = nouns_lang_df[nouns_lang_df['ASSOCIATION/GRAMMATICAL GENDER'] == noun_gender]['WORD'].tolist()
            for noun in nouns:
                for adjective in adjectives:
                    similarity = cossim(model.get_word_vector(noun), model.get_word_vector(adjective))
                    experimental_test_data.append({
                        'LANGUAGE': lang_code,
                        'GRAMMATICAL GENDER OF NOUN': noun_gender,
                        'NOUN': noun,
                        'ADJECTIVE': adjective,
                        'COSINE SIMILARITY': similarity,
                        'GENDER ASSOCIATION OF ADJECTIVE': adj_association
                    })

    combined_data = pd.DataFrame(experimental_test_data)
    
    if use_groupby:
        combined_data = combined_data.groupby(['LANGUAGE', 'GRAMMATICAL GENDER OF NOUN', 'NOUN', 'GENDER ASSOCIATION OF ADJECTIVE'])['COSINE SIMILARITY'].mean().reset_index()

    return combined_data

def remove_adjective_duplicates(lang_code, columns):
    """
    Removes duplicates between masculine and feminine adjective lists for a given language,
    retaining only the form with the higher score. Specifically handles Spanish adjectives
    by comparing masculine and feminine forms and removing the form with the lower score.
    For non-Spanish or non-gender-inflected languages, it removes common adjectives
    based on comparison scores.

    Args:
        lang_code (str): The ISO code for the language, used to load the appropriate CSV files.
        columns (dict): A dictionary specifying the score columns to use for comparison,
                        with keys 'masculine' and 'feminine'.

    This function updates the CSV files by removing the lower-scored duplicates.
    """

    # Load CSV files for masculine and feminine adjectives
    masculine_csv = f'../materials/adjectives/{lang_code}_masculine_adjectives.csv'
    feminine_csv = f'../materials/adjectives/{lang_code}_feminine_adjectives.csv'
    df_masculine = pd.read_csv(masculine_csv)
    df_feminine = pd.read_csv(feminine_csv)

    # Handle Spanish adjectives with gender-specific endings
    if lang_code == 'es':
        for adj in df_masculine['Adjective']:
            if adj.endswith('o'):  # Identifying masculine form
                adj_root = adj[:-1]  # Removing gender-specific ending
                feminine_form = adj_root + 'a'  # Constructing feminine form
                # Check if the feminine form exists in the feminine adjectives DataFrame
                if feminine_form in df_feminine['Adjective'].values:
                    # Compare scores between masculine and feminine forms
                    masculine_score = df_masculine.loc[df_masculine['Adjective'] == adj, columns['masculine']].values[0]
                    feminine_score = df_feminine.loc[df_feminine['Adjective'] == feminine_form, columns['feminine']].values[0]
                    # Remove the form with the lower score
                    if masculine_score > feminine_score:
                        df_feminine = df_feminine[df_feminine['Adjective'] != feminine_form]
                    elif feminine_score > masculine_score:
                        df_masculine = df_masculine[df_masculine['Adjective'] != adj]

    # Handle duplicates in non-Spanish languages or non-gender inflected adjectives
    common_adjectives = set(df_masculine['Adjective']).intersection(df_feminine['Adjective'])
    for adj in common_adjectives:
        # Compare scores for common adjectives
        masculine_score = df_masculine.loc[df_masculine['Adjective'] == adj, columns['masculine']].values[0]
        feminine_score = df_feminine.loc[df_feminine['Adjective'] == adj, columns['feminine']].values[0]
        # Remove the form with the lower score
        if masculine_score > feminine_score:
            df_feminine = df_feminine[df_feminine['Adjective'] != adj]
        elif feminine_score > masculine_score:
            df_masculine = df_masculine[df_masculine['Adjective'] != adj]

    # Save the updated CSV files back to their respective locations
    df_masculine.to_csv(masculine_csv, index=False)
    df_feminine.to_csv(feminine_csv, index=False)
    log(f"Updated CSV files for {lang_code}: removed duplicates with lower scores.")

def create_adjective_review_csv(parquet_file):
    """
    Converts a parquet file containing adjectives into a CSV file with additional information
    including definitions and similarity scores. The output is tailored based on the language
    and gender association of the adjectives.

    Args:
        parquet_file (str): The path to the input parquet file containing adjectives.
    
    This function reads the specified parquet file, adds a 'Definition' column by looking up
    each adjective's definition, sorts the data based on a gender-specific similarity score,
    and saves the resulting DataFrame to a CSV file in a designated directory.
    """

    # Extract language code and gender from the filename
    file_name = os.path.basename(parquet_file)
    language_code, gender = file_name.split('_')[:2]
    log(f'Creating review sheet for language: {language_code}')

    # Load the parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)

    # Add a column with definitions for each adjective
    df['Definition'] = df['Adjective'].apply(lambda x: find_adjective_definition(x))

    # Identify the columns for similarity scores based on gender
    similarity_score_col = f"{gender.lower()}_similarity"
    score_col = f"{gender.lower()}_score"

    # Sort the DataFrame based on the specified similarity score column
    df_sorted = df.sort_values(similarity_score_col, ascending=False)

    # Define the path where the CSV file will be saved
    reviews_dir = '../materials/adjectives'
    csv_file_path = os.path.join(reviews_dir, file_name.replace('.parquet', '.csv'))

    # Save the sorted DataFrame to a CSV file
    df_sorted.to_csv(csv_file_path, index=False)
    log(f"File saved as {csv_file_path}")

def find_minimum_length(languages, genders, unallowed_words):
    """
    Finds the minimum length (number of entries) among various adjective lists across languages and genders,
    excluding specified unallowed words.

    This function iterates over a set of languages and genders, loading corresponding adjective lists from
    parquet files. It filters out any unallowed words from these lists and determines the minimum length
    among them.

    Args:
        languages (list): A list of language codes (str) to be processed.
        genders (list): A list of genders (str) corresponding to the adjective lists to be processed.
        unallowed_words (list): A list of words (str) that are to be excluded from the analysis.

    Returns:
        int: The minimum length (number of valid entries) found among the processed adjective lists.
    """
    min_length = float('inf')  # Initialize with infinity

    # Iterate over each language and gender to process corresponding adjective lists
    for lang_code in languages:
        for gender in genders:
            # Construct the file path for the parquet file
            parquet_file = f'../materials/adjectives/{lang_code}_{gender}_adjectives.parquet'
            # Load the parquet file into a DataFrame
            df = pd.read_parquet(parquet_file)
            # Exclude unallowed words from the DataFrame
            df_filtered = df[~df['Adjective'].isin(unallowed_words)]
            # Update min_length if the length of the current DataFrame is smaller
            min_length = min(min_length, len(df_filtered))

    return min_length

def remove_unwanted_adjectives(csv_file, allowed_words, unallowed_words, markers, min_length, gender):
    """
    Processes an adjective list CSV file by removing unallowed words, filtering out words with specific markers,
    and trimming the list to a specified minimum length based on gender-specific scores.

    Args:
        csv_file (str): Path to the CSV file containing the adjective list.
        allowed_words (set): A set of words that are explicitly allowed, even if they contain unwanted markers.
        unallowed_words (set): A set of words that are explicitly disallowed and should be removed from the list.
        markers (list): A list of substrings that, if found within an adjective's definition, mark the adjective as unwanted.
        min_length (int): The number of top-scoring adjectives to retain in the list after processing.
        gender (str): The gender association of the adjectives, used to determine the score column for sorting.

    Returns:
        pd.DataFrame: The processed DataFrame after filtering and trimming operations.
    """

    # Load data from the specified CSV file into a DataFrame.
    csv_df = pd.read_csv(csv_file)

    # Extract the language code from the filename, assuming it follows a specific naming convention.
    lang_code = csv_file.split('/')[-1][:2]

    # Ensure the 'Adjective' column is treated as strings for consistent processing.
    csv_df['Adjective'] = csv_df['Adjective'].astype(str)

    # Special preprocessing step for Spanish adjectives to handle gender variations.
    if lang_code == 'es':
        processed_unallowed = set()
        for word in unallowed_words:
            # For gendered words, add the root to the set of unallowed words.
            if word.endswith('o') or word.endswith('a'):
                processed_unallowed.add(word[:-1])
            else:
                processed_unallowed.add(word)
        unallowed_words = processed_unallowed

    # Filter out unallowed words by comparing roots for gendered words.
    csv_df = csv_df[~csv_df['Adjective'].apply(lambda x: x[:-1] if (x.endswith('o') or x.endswith('a')) else x).isin(unallowed_words)]

    # Further filter the DataFrame to remove adjectives with unwanted markers in their definitions,
    # unless they are explicitly listed as allowed.
    csv_df = csv_df[(~csv_df['Definition'].apply(lambda x: any(marker in x for marker in markers)) | csv_df['Adjective'].isin(allowed_words))]

    # Sort the DataFrame by score (assuming a specific column naming convention) and trim to the desired length.
    score_col = f"{gender}_score"  # This assumes a specific naming convention for score columns.
    csv_df = csv_df.sort_values(by=score_col, ascending=False).head(min_length)

    # Save the processed DataFrame back to both CSV and Parquet formats.
    csv_df.to_csv(csv_file, index=False)
    parquet_file = csv_file.replace('.csv', '.parquet')
    csv_df.to_parquet(parquet_file, index=False)

    log(f"Updated DataFrame saved as {csv_file} and {parquet_file}")

    return csv_df

def create_adjective_stimulus_files(csv_file):
    """
    Copies an adjective list CSV file to a designated directory for stimulus files.

    This function is designed to prepare stimulus files for experiments or further processing
    by copying the specified CSV file containing an adjective list into a target directory.
    It ensures that the target directory exists before copying and retains the original file name
    in the target location.

    Args:
        csv_file (str): Path to the source CSV file containing the adjective list.
    """
    # Define the directory where the stimulus files will be stored
    target_dir = '../materials/adjectives/stimulus_files'
    # Ensure the target directory exists, creating it if necessary
    os.makedirs(target_dir, exist_ok=True)

    # Construct the path for the target file within the stimulus directory
    target_file_path = os.path.join(target_dir, os.path.basename(csv_file))

    # Copy the CSV file to the target directory
    shutil.copyfile(csv_file, target_file_path)
    log(f"Copied {csv_file} to {target_file_path}")

################################################################
#### bring it all together
################################################################
def run(
        procedures, 
        nouns_file, 
        adjective_gender_association_method, 
        top_n_adjectives, 
        load_method, 
        models, 
        semantic_differential_vectors, 
        use_groupby, 
        remove_adjectives_with_markers, 
        unallowed_words, 
        allowed_words
    ):
    """
    Orchestrate the analysis pipeline.

    Args:
        procedures (dict): Control flags for pipeline procedures. Keys:
            - 'load_models' (bool): Load gensim models.
            - 'crawl_wiktionary_adjectives' (bool): Crawl Wiktionary for adjectives.
            - 'calculate_adjective_similarities' (bool): Calculate gender similarity scores.
            - 'select_top_adjectives' (bool): Select top N gendered adjectives.
            - 'remove_adjective_duplicates' (bool): Remove duplicate adjectives.
            - 'adjective_definition_review' (bool): Generate adjective review CSVs.
            - 'remove_unwanted_adjectives' (bool): Remove unwanted adjectives.
            - 'create_stimulus_files' (bool): Create adjective stimulus files.
            - 'conduct_control_tests' (bool): Conduct control tests.
            - 'conduct_experimental_tests' (bool): Conduct experimental tests.
        nouns_file (str): Path to the noun dataset CSV file.
        adjective_gender_association_method (str): Method for calculating gender associations.
            - 'cosine_similarity': Calculate raw cosine similarity between adjective and gender vectors.
            - 'semantic_differential': Calculate difference between adjective-gender similarity scores.
        top_n_adjectives (int): Number of top gendered adjectives to select per gender.
        load_method (str): Model loading method. Options: 'normal', 'facebook'.
        models (dict): Pre-loaded gensim models keyed by language code.
        semantic_differential_vectors (str): Word vector pairs for semantic differential calculations.
            - 'gender1-gender2': gender1 vector minus gender2 vector (e.g., masculine - feminine).
            - 'gender-person': gender vector minus person vector (e.g., masculine - person).
            - 'gender-Gender': gender vector minus depersonalized Gender vector (e.g., masculine - femininity).
        use_groupby (bool): For experimental tests, group adjectives for a given noun into one average cosine similarity, so instead of n(nouns)*n(adjectives) data points, you only have n(nouns) datapoints. Best for strip plots to see individual points.
        remove_adjectives_with_markers (list): Substrings from Wiktionary definitions marking adjectives for removal.
        unallowed_words (set): Words to exclude from final adjective lists.
        allowed_words (set): Words to retain regardless of removal markers.

    Recommended Use:
        run(
        procedures={
            'load_models':True,

            # Crawling wiktionary nouns is not yet implemented. We did not use this method to gather our nouns.
            'crawl_wiktionary_nouns':False,
            # We have already gathered adjectives from Wiktionary.
            'crawl_wiktionary_adjectives':False,

            'calculate_adjective_similarities':True,

            'select_top_adjectives':True,

            'remove_adjective_duplicates':True,
            'adjective_definition_review':True,
            'remove_unwanted_adjectives':True,

            'create_stimulus_files':True,

            'conduct_control_tests':True,
            'conduct_experimental_tests':True,
        },
        models=models,
        load_method='normal',
        # normal, facebook
        nouns_file='../materials/nouns.csv',
        top_n_adjectives=100,
        adjective_gender_association_method='cosine_similarity',
        use_groupby=True,
        remove_adjectives_with_markers=["dated", "archaic", "dialectal", "rare", "ordinal number", "obsolete", "offensive"],
        semantic_differential_vectors='gender1-gender2',
        unallowed_words=['lesb', 'debonair', 'vestal', 'sunamita', 'negrid', 'Brummagem', 'follable', 'untervögelt', 'schasaugert', 'Emeser', 'fünfhundertste', 'Poppersch', 'Schlänger', 'Römer', 'Latina', 'titless', 'pussy', 'foine', 'mosuo', 'fáustico', 'indio', 'rixig', 'hiborio', 'abgeschmack', 'kaki', 'klaviform', 'TK', 'antimalthusianisch', 'Danubian', 'eblaitisch', 'elfminütig', 'Fregesch', 'jakobinisch', 'Malthusianisch', 'meißenisch', 'neunminütig', 'rahn', 'vierzigminütig', 'zwölfminütig', 'Afro-Latina', 'Dianic', 'Filipina', 'lady-like', 'MAAB', 'menstruate', 'obstetrical', 'Quebecoise', 'Rubenesque', 'woman-centric', 'vinny', 'twinky', 'Welshy', 'turrible', 'mick', 'fooking', 'particuler', 'legendry', 'awsome', 'roy', 'neo-Hegelian', 'phun', 'niiice', 'Democritean', 'Hegelian', 'Rothbardian', 'gent', 'afrodescendiente', 'axumita', 'curvi', 'delhita', 'feminazi', 'madrense', 'mizrají', 'oseta', 'postparto', 'sefaradita', 'sefardí', 'sefardita', 'transgenerista', 'fuckin', 'hanbalitisch', 'antimalthusianisch', 'Malthusianisch', 'antimalthusianisch', 'malthusisch','gustiös', 'hanbalitisch','handgehoben','scheiß', 'sturm', 'terrisch','Madonna-like','smoove', 'tuff','hench','insano', 'mofo', 'cutty', 'piff', 'jake', 'propa', 'mank','LGBT','papaya', 'child-bearing', 'plus-sized', 'post-partum', 'vulval', 'ben', 'unpossible', 'antifeminist', 'LGTB', 'LGTBI', 'babylonisch', 'erzgebirgisch', 'hinreissend', 'niedersorbisch', 'Sanct', 'sasanidisch', 'saudisch', 'altniederländisch', 'bohrsch', 'britannisch', 'danubisch', 'dreiundvierzigminütig', 'drittelzahlig', 'etatmässig', 'fünfundzwanzigminütig', 'fünfunddreißigminütig', 'fünfminütig', 'Hitlersch', 'koblenzisch', 'Luthersch', 'sechzigminütig', 'südatlantisch', 'Cesarean', 'prochoice', 'almight', 'cock-sure', 'cooool', 'nooby', 'peart', 'phantastic', 'Smithian', 'barakaldarra', 'bartorosellista', 'cefeida', 'chilota', 'dailamita', 'estambulita', 'kábila', 'mazahua', 'ondarrutarra', 'ranjana', 'helle', 'zirkummediterran', 'südatlantisch', 'preggers', 'vajazzled', 'shite', 'steezy', 'tinhorn', 'widdly', 'afrotropical', 'apollardado', 'mijita', 'ladilla', 'gray-haired', 'heavy-set', 'middleaged', 'Jew,' 'moustached', 'African-American', 'childbearing', 'Filipina', 'Madonna-like', 'newly-wed', 'Shunamite', 'Syrophoenician', 'teen-age', 'teen-aged', 'transgendered', 'grown-ass', 'mustached', 'Caucasian', 'biracial', 'mixed-race', 'thirties', 'forties', 'clean-shaved', 'moustachioed', 'dark-skinned', 'teenaged', 'mustachioed', 'ape', 'Afroestadounidense', 'Birracial', 'Indígena', 'sexi', 'Extraconyugal', 'Israelita', 'untrew', 'cristiano', 'jóven', 'afroestadounidense', 'birracial', 'Emesener', 'baktrisch', 'israelita', '♥-lich', 'vierzigmonatig', 'währschaft', 'wolgadeutsch', 'amisch', 'dreiundfünfzigjährig', 'kraftwerkisch', 'malisch', 'Palmyrer', 'Portaner' ,'achtundvierzigmonatig', 'padre', 'hypoäolisch', 'schwatt', 'sechsunddreißigmonatig', 'israelita', 'dreißigmonatig', 'einunddreißigeckig', 'schwul', 'mannmännlich'],
        allowed_words=['rascal', 'transexual'],
    )
    """
    nouns_df = load_dataframe(nouns_file)
    min_length = top_n_adjectives
    if adjective_gender_association_method == 'cosine_similarity':
        for gender in columns.keys(): columns[gender] = f'{gender}_similarity'
    elif adjective_gender_association_method == 'semantic_differential':
        for gender in columns.keys(): columns[gender] = f'{gender}_score'
    if procedures['load_models']:
        # Load models for each language
        models = {lang: load_model(lang, load_method) for lang in languages}

    if procedures['crawl_wiktionary_adjectives']:
        # Adjectives extraction and saving
        adjectives_url = f'https://en.wiktionary.org/wiki/Category:{lang_data["full_name"]}_adjectives'
        adjectives = extract_adjectives(lang_code, adjectives_url)
        save_adjectives_to_parquet(adjectives, lang_code, f'adjectives/{lang_code}_adjectives.parquet')

    if procedures['calculate_adjective_similarities']:
        # Populate adjective list with gender similarity data
        for lang_code in languages.keys():
            log(f"Performing gender similarity calculations for {languages[lang_code]['full_name']}...")
            calculate_adjective_similarities(lang_code)
            log(f"Calculations completed for {languages[lang_code]['full_name']}.")

    if procedures['select_top_adjectives']:
        # Select the top n most masculine or feminine adjectives
        for lang_code in languages.keys():
            masculine, feminine = select_top_words(lang_code, num_rows=top_n_adjectives, method=adjective_gender_association_method, semantic_differential_vectors=semantic_differential_vectors)
            if lang_code == 'es':
                masculine, feminine = duplicate_spanish_adjectives(masculine, 'masculine'), duplicate_spanish_adjectives(feminine, 'feminine')
            log(f"Selected top adjectives for {languages[lang_code]['full_name']}: Masculine: {len(masculine)}, Feminine: {len(feminine)}")
    
    if procedures['remove_adjective_duplicates']:
        for lang_code in languages.keys():
            remove_adjective_duplicates(lang_code, columns)
        
    # Turn Parquet files into csv files for manual inspection or readability during communication
    if procedures['adjective_definition_review']:
        for lang_code in languages.keys():
            for gender in targets:
                create_adjective_review_csv(f'../materials/adjectives/{lang_code}_{gender}_adjectives.parquet')

    if procedures['remove_unwanted_adjectives']:
        for lang_code in ['en', 'es', 'de']:
            for gender in ['masculine', 'feminine']:
                csv_file = f'../materials/adjectives/{lang_code}_{gender}_adjectives.csv'
                parquet_file = f'../materials/adjectives/{lang_code}_{gender}_adjectives.parquet'
                min_length = find_minimum_length(languages, targets, unallowed_words)
                remove_unwanted_adjectives(csv_file, allowed_words, unallowed_words, remove_adjectives_with_markers, min_length, gender)

    if procedures['create_stimulus_files']:
        min_length = find_minimum_length(languages, targets, unallowed_words)

        for lang_code in languages:
            for gender in targets:
                csv_file = f'../materials/adjectives/{lang_code}_{gender}_adjectives.csv'
                create_adjective_stimulus_files(csv_file)

    # Iterate through each language
    for lang_code, lang_data in languages.items():
        log(f"Processing language: {lang_data['full_name']}")
        model = models[lang_code]

        # Conduct Control Tests
        if procedures['conduct_control_tests']:
            log(f"Conducting control tests for {lang_data['full_name']}")
            control_data_dir = f'../data/embeddings/control'
            os.makedirs(control_data_dir, exist_ok=True)

            control_data = create_control_test_dataframe(lang_code, nouns_df, model)
            control_data.to_parquet(f'{control_data_dir}/{lang_code}_control_data.parquet')

            for adj_association in targets:
                control_data = create_control_test_dataframe(lang_code, nouns_df, model)
                control_data.to_parquet(f'{control_data_dir}/{lang_code}_control_data.parquet')

        # Conduct Experimental Tests
        if procedures['conduct_experimental_tests']:
            
            log(f"Conducting experimental tests for {lang_data['full_name']}")
            test_data_dir = f'../data/embeddings/experimental'
            os.makedirs(test_data_dir, exist_ok=True)

            experimental_data = create_experimental_test_dataframe(lang_code, nouns_df, model, use_groupby)
            experimental_data.to_parquet(f'{test_data_dir}/{lang_code}_test_data.parquet')

    log("Finished processing for all languages.")