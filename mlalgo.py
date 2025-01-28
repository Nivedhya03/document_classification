import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import hstack
import os
import numpy as np


# Step 1: Load and parse TF-IDF files
def load_tfidf_files(file_paths, labels):
    datasets = []
    for file_path, label in zip(file_paths, labels):
        data = pd.read_csv(file_path, delimiter='\t')  # Adjust delimiter as needed
        data['label'] = label
        if 'text' not in data.columns:
            data.rename(columns={data.columns[0]: 'text'}, inplace=True)
        datasets.append(data)
    return pd.concat(datasets, ignore_index=True)


# Step 2: Add domain-specific features (keywords)
def add_keyword_features(text_data):
    keywords = {
        'warranty_deed': ['guarantee', 'warranty', 'obligation', 'repair'],
        'deed_of_satisfaction': ['satisfaction', 'release', 'debt', 'creditor'],
        'promissory_note': ['loan', 'borrower', 'lender', 'credit'],
        'deed_of_trust': ['trust', 'security', 'property', 'guarantor']
    }
    features = []
    for doc in text_data:
        keyword_features = {key: sum([doc.count(word) for word in words]) for key, words in keywords.items()}
        features.append(list(keyword_features.values()))
    return features


# Step 3: Prepare Features and Labels
def prepare_features_and_labels(dataset, text_column='text', label_column='label'):
    X = dataset[text_column]
    y = dataset[label_column]
    return X, y


# Step 4: Train-Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# Step 5: Train the Model
def train_model(X_train, y_train):
    model = SVC(probability=True)  # Enable probability estimation (for predict_proba)
    model.fit(X_train, y_train)
    return model


# Step 6: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


# Step 7: Preprocess a New Document
def preprocess_new_document(file_path, vectorizer):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith('.pdf'):
        raise ValueError("This function currently does not support PDFs. Convert to .txt first.")

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return vectorizer.transform([text])


# Step 8: Classify New Documents
def classify_new_document(model, new_tfidf_vector):
    return model.predict(new_tfidf_vector)


# Beautified Printing of Classification Report
def pretty_print_classification_report(report):
    print("\nClassification Report (Precision, Recall, F1-Score):")

    # Print the headers
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    print(f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<10} {headers[3]:<10} {headers[4]:<8}")
    print("-" * 60)

    # Print each class's results in a neat, aligned format
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:  # Skip summary rows
            print(
                f"{label.capitalize():<20} {metrics['precision']:<12.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<8.1f}")

    # Print the overall performance metrics
    print("-" * 60)
    print(f"{'Accuracy':<20} {report['accuracy']:<12.3f}")
    print(
        f"{'Macro avg':<20} {report['macro avg']['precision']:<12.3f} {report['macro avg']['recall']:<10.3f} {report['macro avg']['f1-score']:<10.3f}")
    print(
        f"{'Weighted avg':<20} {report['weighted avg']['precision']:<12.3f} {report['weighted avg']['recall']:<10.3f} {report['weighted avg']['f1-score']:<10.3f}")


# Main execution
def main():
    # File paths and corresponding labels for training documents
    file_paths = [
        r"C:\Users\ACER\Desktop\doc seperation\deed_of_satisfaction.txt",
        r"C:\Users\ACER\Desktop\doc seperation\deed_of_trust.txt",
        r"C:\Users\ACER\Desktop\doc seperation\promissory_note.txt",
        r"C:\Users\ACER\Desktop\doc seperation\warenty_deed.txt"
    ]
    labels = ['deed_of_satisfaction', 'deed_of_trust', 'promissory_note', 'warranty_deed']

    # Load and prepare data
    dataset = load_tfidf_files(file_paths, labels)

    # Balance dataset if needed
    min_samples_per_class = 10  # Ensure at least 10 samples per class
    balanced_dataset = pd.concat(
        [
            resample(
                dataset[dataset['label'] == label],
                replace=True,
                n_samples=min_samples_per_class,
                random_state=42
            )
            for label in dataset['label'].unique()
        ]
    )

    # Prepare features and labels
    X, y = prepare_features_and_labels(balanced_dataset, text_column='text', label_column='label')

    # Vectorize text data using n-grams (unigrams + bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use both unigrams and bigrams
    X_tfidf = vectorizer.fit_transform(X)

    # Add domain-specific keyword features
    keyword_features = add_keyword_features(X)
    X_combined = hstack([X_tfidf, keyword_features])  # Combine both sets of features

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_combined, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    report = evaluate_model(model, X_test, y_test)

    # Pretty print the classification report
    pretty_print_classification_report(report)

    # **New Document Classification**:
    # Specify the path of the new document you want to classify
    new_document_path = r

    try:
        # Preprocess the new document using the same vectorizer and add keyword features
        new_document_tfidf = preprocess_new_document(new_document_path, vectorizer)
        new_document_keywords = add_keyword_features([new_document_path])  # Add keyword features
        new_document_combined = hstack([new_document_tfidf, new_document_keywords])  # Combine features

        # Get probabilities for all classes (instead of just predicting the class)
        probabilities = model.predict_proba(new_document_combined)

        # Get the top two predicted classes based on probability
        top_two_classes = np.argsort(probabilities[0])[-2:][::-1]  # Indices of the top two classes
        predicted_labels = [model.classes_[i] for i in top_two_classes]

        print(f"Top 2 predicted classes: {predicted_labels}")

        # If two classes have similar probabilities, compare them using keywords
        if probabilities[0][top_two_classes[0]] == probabilities[0][top_two_classes[1]]:
            class1 = predicted_labels[0]
            class2 = predicted_labels[1]
            comparison_result = compare_by_keywords(class1, class2, keyword_features)
            print(f"Final predicted class based on keyword comparison: {comparison_result}")
        else:
            print(f"Predicted class: {predicted_labels[0]}")  # Select the class with the highest probability

    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
