# Fortune 500 NLP-Powered QA System

An interactive question-answering system built using [SentenceTransformers](https://www.sbert.net/) and semantic similarity to allow users to query structured data from the Fortune 500 Companies dataset.

Developed by **Almohtadey Metwaly**.

---

##  Overview

This project loads a dataset of Fortune 500 companies and uses sentence embeddings to semantically answer natural language questions like:

- "What is the revenue of Walmart?"
- "Where is the headquarters of ExxonMobil?"
- "Which company was a newcomer?"
- "What industry does Apple belong to?"

The system builds a knowledge base of human-readable sentences and matches user questions via cosine similarity.

---

##  Features

- Semantic search over structured data using **Sentence-BERT (all-MiniLM-L6-v2)**
- Interactive command-line interface (CLI)
- Model evaluation on a test set
- Preprocessing and cleaning of numeric and textual fields
- Handles missing data gracefully

---

##  Dataset

- **File Used**: `Fortune 500 Companies.csv`
- Columns used:
  - `name`, `rank`, `year`, `industry`, `revenue_mil`, `profit_mil`, `headquarters_city`, `headquarters_state`, `newcomer_to_fortune_500`

Make sure this file exists at the path specified in the code or change the `file_path` variable.

## Setup and Installation

1. Install dependencies:
   ```
   pip install pandas sentence-transformers scikit-learn numpy
   ```

2. Place your CSV file at the path specified in the code or change it 
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
