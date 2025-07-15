import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Load Dataset
file_path = r"C:\Users\145989\Downloads\Fortune 500 Companies.csv"
df = pd.read_csv(file_path)
df.rename(columns={'year of  Rank': 'year'}, inplace=True)
df.fillna('-', inplace=True)

for col in ['market_value_mil', 'revenue_mil', 'profit_mil', 'asset_mil', 'employees']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Load SentenceTransformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate Knowledge Base Sentences
def generate_knowledge_sentence(row):
    return (f"{row['name']} was ranked {int(row['rank'])} in {int(row['year'])}. "
            f"It operates in the {row['industry']} sector. Revenue was ${row['revenue_mil']:,.1f}M, "
            f"profit ${row['profit_mil']:,.1f}M. Headquartered in {row['headquarters_city']}, {row['headquarters_state']}. "
            f"{'The company was a newcomer that year.' if row['newcomer_to_fortune_500'] != '-' else ''}")

knowledge_sentences = [generate_knowledge_sentence(row) for _, row in df.iterrows()]
sentence_embeddings = model.encode(knowledge_sentences, convert_to_tensor=True)

# Function to Query the Knowledge Base
def get_top_results(user_question, top_k=5):
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
    top_results = np.argsort(-scores)[:top_k]
    return top_results, scores

def answer_question(user_question, top_k=5):
    print("\n🔍 Searching for answers...\n")
    top_results, scores = get_top_results(user_question, top_k)
    for idx in top_results:
        print(f"→ {knowledge_sentences[idx]} (Score: {scores[idx].item():.4f})\n")

# Function to Evaluate the Model
def evaluate_model(test_questions, test_answers, top_k=1):
    true_positives = 0
    predictions = []
    actuals = []

    for question, answer_index in zip(test_questions, test_answers):
        top_results, _ = get_top_results(question, top_k)
        if answer_index in top_results:
            predictions.append(1)
        else:
            predictions.append(0)
        actuals.append(1)  # All are assumed relevant

    precision = precision_score(actuals, predictions, zero_division=0)
    recall = precision
    f1 = f1_score(actuals, predictions, zero_division=0)
    accuracy = accuracy_score(actuals, predictions)

    print("\n=== 📊 Model Evaluation ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

# Interactive CLI
def main():
    print("=== 🧠 Fortune 500 NLP-Powered QA System ===")

    while True:
        print("\nOptions:\n1. Ask a question\n2. Evaluate model\n3. Exit")
        choice = input("Choose an option (1/2/3):\n> ")

        if choice == '1':
            query = input("\nAsk your question:\n> ")
            answer_question(query)
        elif choice == '2':
            # Example Test Data (modify as needed)
            test_questions = [
                "What is the revenue of Walmart in a specific year?",
                "Where is the headquarters of ExxonMobil?",
                "Which company was a newcomer?",
                "What is the industry of Apple?",
                "What is the rank of Berkshire Hathaway in a specific year?",
            ]
            # Manually set the correct answer indices based on your data
            test_answers = [0, 1, 2, 3, 4]  # Replace with correct indices
            evaluate_model(test_questions, test_answers)
        elif choice == '3':
            print("Goodbye! 👋")
            break
        else:
            print("Invalid option. Please try again.")

# Run the app
if __name__ == '__main__':
    main()