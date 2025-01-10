import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from transformers import AutoTokenizer
import openai
openai.api_key_path = r"C:\Users\danie\Documents\GitHub\NLP_Project_AI23\API_KEY_OPENAI_250110.txt"

def create_checked_texts(array_text, tokenizer):
    """Function tokenizes the input (X_test) and returns a checked text and sentiment result.
    If number of tokens > 512, the function calls OpenAI API to summarize the
    text and tokenize the summary."""
    checked_texts = pd.Series(dtype=str)
    nbr_API_calls = 0
    for i, string in enumerate(array_text):
        try:
            tokenized_text = tokenizer(string, padding=True, truncation=False, return_tensors="tf")
        except Exception as e:
            print(f"Error: {e}")
            return None
        
        length = len(tokenized_text['input_ids'][0])
        if length <= 512:
            checked_text = string
        else:
            checked_text = summarize_text(string)
            nbr_API_calls += 1
        
        checked_texts.loc[i] = checked_text

    return checked_texts, nbr_API_calls

def create_sentiment_analysis(array_text, sentiment_pipeline):
    """Function creates sentiment analysis for an array of strings."""
    sentiment_results = pd.Series(dtype=str)
    for i, string in enumerate(array_text):
        sentiment_results.loc[i] = sentiment_pipeline(string)[0]['label']
    sentiment_results = sentiment_results.str.lower()
    return sentiment_results


def plot_confusion_matrix(cm):
    """Function plots a confusion matrix."""
    plt.figure(figsize=(7, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize this text: {text}"},
            ],
        max_tokens=500,  # Limit the output to 500 tokens
        temperature=0.5  # Controls randomness, 0.5 is a balanced value
    )
    
    # Extract the summary from the response
    summary = response['choices'][0]['message']['content']
    return summary