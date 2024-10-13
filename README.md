# Plagiarism Checker v1

A Python-based **Plagiarism Checker** that uses **Natural Language Processing (NLP)** techniques to detect textual plagiarism by comparing the similarity between different documents. This project processes large volumes of text to determine how similar one document is to another, making it useful for academic and professional use cases.

## Features
- **Text Comparison**: Compares multiple documents and identifies the percentage of similarity between them.
- **Preprocessing**: Tokenizes, stems, and removes stopwords from the text to standardize the input before comparing.
- **NLP Techniques**: Utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to measure how closely related two documents are.
- **Jupyter Notebook Implementation**: The main logic is implemented in a Jupyter Notebook for easy experimentation and result visualization.


## Installation

### Prerequisites
- Python 3.x
- Libraries: The required Python packages can be installed using `requirements.txt`.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/anushkasrivastava1303/plagarismchecker_v1.git
   cd plagarismchecker_v1
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the **start.ipynb** file in Jupyter Notebook:
   ```bash
   jupyter notebook start.ipynb
   ```

## Usage
1. Open **start.ipynb** in Jupyter Notebook.
2. Load the documents (e.g., `animals.txt`, `life.txt`, `livingbeing.txt`) that you want to compare for plagiarism.
3. Run the cells in the notebook to preprocess the text and check for similarities between the documents.
4. The output will show the percentage similarity between each pair of documents.

### Example
Here is an example of how to compare two documents:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the documents
with open('life.txt', 'r') as file1, open('livingorganism.txt', 'r') as file2:
    doc1 = file1.read()
    doc2 = file2.read()

# Preprocess and vectorize
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

# Compute cosine similarity
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Similarity between documents: {similarity[0][0] * 100:.2f}%")
```

## How it Works
1. **Text Preprocessing**: The input documents are tokenized, and stopwords are removed. Optionally, stemming or lemmatization can be applied to standardize the text.
2. **TF-IDF Calculation**: Each document is vectorized using **TF-IDF** to convert the text into numerical form based on word frequency.
3. **Cosine Similarity**: The similarity between two documents is calculated using **cosine similarity**, which measures the cosine of the angle between the vector representations of the documents.
4. **Output**: The output is a percentage score indicating how similar one document is to another.

## Technologies Used
- **Python**: The main language used to implement the plagiarism checker.
- **Jupyter Notebook**: For running and experimenting with the plagiarism checking process.
- **Scikit-learn**: For TF-IDF vectorization and similarity measurements.
- **NLP Techniques**: Tokenization, stopword removal, TF-IDF, and cosine similarity.

## Project Structure
```
plagarismchecker_v1/
├── start.ipynb           # Main Jupyter notebook for the plagiarism checker
├── README.md             # Project documentation
├── animals.txt           # Sample document
├── life.txt              # Sample document
├── livingbeing.txt       # Sample document
├── livingmatter.txt      # Sample document
├── livingorganism.txt    # Sample document
```

## Future Improvements
- **Synonym Handling**: Add support for detecting similarities even when different words (synonyms) are used.
- **Improved Accuracy**: Use more advanced NLP models like **BERT** or **Doc2Vec** to improve similarity detection.
- **UI Development**: Develop a simple web-based UI for uploading documents and viewing plagiarism results.
- **Large Dataset Support**: Improve scalability to handle larger datasets or multiple document comparisons at once.

## Contributing
Contributions are welcome! If you'd like to improve this project or fix bugs, feel free to submit a pull request or open an issue.
