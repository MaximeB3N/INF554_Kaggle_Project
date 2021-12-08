import io
import json
import re
from tqdm import tqdm
import numpy as np
import string


def get_id(file_str):
    """"
    Get the id of the abstract
    """
    return int(file_str.split("----")[0])

def get_json(file_str):
    """
    Get the json file of the abstract
    """
    new_file_str = file_str.replace("-","")
    json_str = new_file_str.split(str(get_id(file_str)))[-1]

    return json.loads(json_str)

def get_descritpion(file_str):
    """
    Get the description of the abstract
    """
    json_file = get_json(file_str)
    words = [""]*int(json_file["IndexLength"])

    for word in json_file["InvertedIndex"].keys():
        indexes = json_file["InvertedIndex"][word]

        for index in indexes:
            words[int(index)] = word
        
        
    return words #' '.join(words).replace("\n", " ")


def get_line(file_str):
    """
    Get the line of the abstract
    """
    words = get_descritpion(file_str)

    return ' '.join(words).replace("\n", "")

def clean_text(text, tokenizer, stopwords):
    """Pre-process text and generate tokens

    Args:
        text: Text to tokenize.

    Returns:
        Tokenized text.
    """
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation

    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens


def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in tqdm(list_of_docs):
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

def open_file(path):
    """
    Open a file and return the content
    """
    print("Loading of every abstracts...")
    f =  open(path, "r", encoding="utf-8")
    output = f.readlines()
    f.close()
    print("finished !")
    return output
