# Fortune 500 NLP-Powered QA System

This is a question-answering system that uses semantic similarity (via SentenceTransformers) to retrieve answers from a structured dataset of Fortune 500 companies.

Developed by: Almohtadey Metwaly

---

## Overview

The project processes a dataset of Fortune 500 companies and creates a knowledge base of human-readable sentences. It then allows users to ask natural language questions and returns the top semantically similar answers.

---

## Features

- Semantic similarity search using Sentence-BERT (all-MiniLM-L6-v2)
- Text preprocessing and numeric normalization
- Handles missing values with default placeholders
- Interactive CLI interface
- Model evaluation with precision, recall, F1-score, and accuracy

---

## Dataset

- File: Fortune 500 Companies.csv
- Key columns used: 
  - name
  - rank
  - year
  - industry
  - revenue_mil
  - profit_mil
  - headquarters_city
  - headquarters_state
  - newcomer_to_fortune_500

---

## Setup and Installation

1. Install dependencies:
   ```
   pip install pandas sentence-transformers scikit-learn numpy
   ```

2. Place your CSV file at the path specified in the code or change it 
   ```


3. Run the script:
   ```
   python your_script_name.py
   ```

---

## CLI Options

- Ask a question and retrieve top-k similar knowledge sentences
- Evaluate the model using test data
- Exit the application

---

## Example

User input:
```
What is the revenue of Walmart?
```

Output:
```
Walmart was ranked 1 in 2020. It operates in the Retail sector. Revenue was $524,000.0M, profit $15,000.0M. Headquartered in Bentonville, AR.
```

---

## Evaluation

To evaluate:
- Modify `test_questions` with your test queries
- Provide `test_answers` as the correct knowledge base indices

The system will compute:
- Precision
- Recall
- F1-score
- Accuracy

---

## Notes

- Retrieval is based on cosine similarity with sentence embeddings.
- This is not a generative QA system.
- The model does not extract fine-grained answers from text.

---

## Author

Almohtadey Metwaly

---

## License

This project is intended for academic and educational use only.
