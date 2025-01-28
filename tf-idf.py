import os
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS  # Import stopwords from spaCy
from docx import Document  # For reading .docx files

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")


# Function to read all Word files from a given folder
def read_word_files(folder_path):
    documents = []
    labels = []  # Labels for each document
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if file_name.endswith('.docx'):  # Only process .docx files
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the content of the .docx file
                doc = Document(file_path)
                doc_text = ' '.join([para.text for para in doc.paragraphs if para.text.strip() != ''])

                # Remove content inside square brackets [ ]
                doc_text = re.sub(r'\[.*?\]', '', doc_text)

                # Add the cleaned document to the list if it’s not too short
                if len(doc_text.strip()) > 10:  # Only keep documents that are reasonably long
                    documents.append(doc_text)
                    label = file_name.split('_')[0]  # e.g., 'label_filename.docx' -> 'label'
                    labels.append(label)

            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    print(f"Total documents read: {len(documents)}")

    return documents, labels


# Function to remove stopwords using spaCy
def remove_stopwords_with_spacy(text):
    doc = nlp(text)
    # Filter out the stopwords
    filtered_tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
    return " ".join(filtered_tokens)


# Function to perform TF-IDF and save the top words to a text file
def perform_tfidf_and_save(documents, top_n=10, output_txt='top_words_tfidf.txt'):
    # Remove stopwords using spaCy
    cleaned_documents = [remove_stopwords_with_spacy(doc) for doc in documents]

    # Initialize the TfidfVectorizer with no stop words
    vectorizer = TfidfVectorizer(stop_words=None)  # No stop words removal here as we already filtered using spaCy
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_documents)
    except ValueError as e:
        print(f"Error during TF-IDF transformation: {e}")
        return {}

    # Get the words and their corresponding TF-IDF values
    feature_names = vectorizer.get_feature_names_out()
    dense_matrix = tfidf_matrix.todense()

    # Calculate mean TF-IDF for each word across all documents
    mean_tfidf = dense_matrix.mean(axis=0).tolist()[0]

    # Create a dictionary of words and their mean TF-IDF score
    word_tfidf_dict = {word: score for word, score in zip(feature_names, mean_tfidf)}

    # Sort the words by TF-IDF score in descending order and select top N
    sorted_words = sorted(word_tfidf_dict.items(), key=lambda item: item[1], reverse=True)

    # Save the top N frequent words to a text file
    with open(output_txt, 'w', encoding='utf-8') as file:
        for word, score in sorted_words[:top_n]:
            file.write(f"{word}: {score}\n")

    return sorted_words[:top_n]

if __name__ == '__main__':
    folder_path = r"C:\Users\ACER\Desktop\doc seperation\warrenty deed"

    # Get the directory of the folder where the documents are stored
    folder_directory = os.path.dirname(folder_path)

    # Define the output text file path (same as the folder of the documents)
    output_txt = os.path.join(folder_directory, 'top_words_tfidf.txt')

    # Step 1: Read all the Word files and labels from the specified folder
    documents, labels = read_word_files(folder_path)  # Corrected function name

    # Step 2: Perform TF-IDF on the documents and get the top frequent words
    top_words = perform_tfidf_and_save(documents, top_n=10, output_txt=output_txt)

    # Output the top words saved to text file
    print(f"Top words have been saved to: {output_txt}")
