import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras import layers
from transformers import pipeline
from transformers import AutoTokenizer
import openai
import time


openai.api_key_path = (
    r"C:\Users\danie\Documents\GitHUb\NLP_Project_AI23\key.txt"
)


def create_checked_texts(array_text, tokenizer):
    """Function tokenizes the input (X_test) and returns a checked text and sentiment result.
    If number of tokens > 512, the function calls OpenAI API to summarize the
    text and tokenize the summary."""
    checked_texts = pd.Series(dtype=str)
    nbr_API_calls = 0
    for i, string in enumerate(array_text):
        try:
            tokenized_text = tokenizer(
                string, padding=True, truncation=False, return_tensors="tf"
            )  # padding=false
        except Exception as e:
            print(f"Error: {e}")
            return None

        length = len(tokenized_text["input_ids"][0])
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
        sentiment_results.loc[i] = sentiment_pipeline(string)[0]["label"]
    sentiment_results = sentiment_results.str.lower()
    return sentiment_results


def create_openai_sentiment_analysis(array_text, model="gpt-3.5-turbo"):
    """Function creates sentiment analysis for an array of strings."""
    nbr_API_calls = 0
    openai_sentiment_results = pd.Series(dtype=str)
    for i, string in enumerate(array_text):
        openai_sentiment_results.loc[i] = openai_sentiment_analysis(string, model=model)
        print(string[0:50])
        print(openai_sentiment_results.loc[i])
        nbr_API_calls += 1
        print(f"API calls: {nbr_API_calls}")
    openai_sentiment_results = openai_sentiment_results.str.lower()
    return openai_sentiment_results, nbr_API_calls


def plot_confusion_matrix(cm, title=None):
    """Function plots a confusion matrix."""
    plt.figure(figsize=(7, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix {title}")
    plt.show()


def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if needed
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text.",
            },
            {"role": "user", "content": f"Summarize this text: {text}"},
        ],
        max_tokens=500,  # Limit the output to 500 tokens
        temperature=0.5,  # Controls randomness, 0.5 is a balanced value
    )

    # Extract the summary from the response
    summary = response["choices"][0]["message"]["content"]
    return summary


def openai_sentiment_analysis(text, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that based on a text creates a sentiment analysis.",
            },
            {
                "role": "user",
                "content": f"Create a sentiment analysis on this text: {text}. The answer from you should be either 'positive' or 'negative'and always lowercase. Never answer with anything else than 'positive' or 'negative'",
            },
        ],
        max_tokens=10,  # Limit the output to 10 tokens
        temperature=0.5,  # Controls randomness, 0.5 is a balanced value
    )

    # Extract the summary from the response
    openai_sentiment_analysis = response["choices"][0]["message"]["content"]
    
    if model == "gpt-4":
        time.sleep(2)  # Sleep for 2 second to avoid API rate limit
    else:
        time.sleep(0.1)

    return openai_sentiment_analysis


def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model
