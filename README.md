# Customer Support Question & Answer GPT

## Overview
This project utilizes the GPT-3 model from OpenAI to create a customer support chatbot that can answer user's questions. It uses the text-embedding-ada-002 model for embedding and the gpt-3.5-turbo model for question answering. The text is indexed using FAISS for efficient retrieval.

## Dataset
Dataset
The dataset used for training the customer support chatbot is the [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). It provides a 
collection of customer support interactions on Twitter, which serves as the basis for training the chatbot to respond to user queries effectively.

## Prerequisites
OpenAI API key: You will need an OpenAI API key to access the GPT-3 model and make API calls.

## Arguments

* **company**: The customer support ID on Twitter. You can find a list of customer support IDs in the `data/companies.csv` file.
* **gpt_model**: The name of the GPT model to use. Refer to the [OpenAI model documentation](https://platform.openai.com/docs/models/gpt-3-5) for available options. The default 
* is gpt-3.5-turbo.
* **embedding_model**: The name of the embedding model to use. The default is `text-embedding-ada-002`.
* **data_path**: The path to the data directory.
* **index_path**: The path to the pickle file that stores the FAISS index.
* **openai_api_key**: Your OpenAI API key.

## How to run
1. Install packages
```python 
pip install requiremetns.txt
```
2. Export your OpenAI API key
```python
export OPENAI_API_KEY='sk-...'
```
3. Run GPT
```python
python main.py --company AppleSupport --openai-api-key [OPENAI_API_KEY]
```
4. Type your question in terminal
```python
What do you want to ask the customer support? (Type 'stop' to quit)
Question: [Your question]
```
