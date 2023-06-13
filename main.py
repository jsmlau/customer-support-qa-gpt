import argparse
from pathlib import Path
import os

from utils import *
from gpt_agent import GPTAgent

FILE = Path(__file__).resolve()  # current file path
ROOT = FILE.parents[0]  # current dir


def run(company: str = 'AppleSupport',
        gpt_model: str = 'gpt-3.5-turbo',
        embedding_model: str = 'text-embedding-ada-002',
        data_path: str = ROOT/'data/twcs.csv',
        index_path: str = ROOT/'data/faiss_index.pkl',
        openai_api_key: str = '',
        verbose: bool = False):

    os.environ['OPENAI_API_KEY'] = openai_api_key
    gpt_agent = GPTAgent()

    # Preprocess data
    df = data_preprocessing(data_path, company).sort_values(
        by='in_response_to_tweet_id', ascending=True)

    # Convert dataframe into conversation threads texts
    texts = get_texts(df)

    # Load/ create embeddings
    db = gpt_agent.load_embeddings(
        texts, embedding_model, index_path)

    # Start
    history = ''

    print('\nWhat do you want to ask the customer support? (Type \'stop\' to quit)')

    while True:
        question = input('Question: ')

        if question == 'stop':
            break

        response, history = gpt_agent.chat(
            question, history, db, gpt_model, verbose)
        print('Support: ', response)


def main(opt):
    run(**vars(opt))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--company', type=str, default='AppleSupport',
                        help='the author id of a company on twitter.')
    parser.add_argument('--gpt-model', type=str, default='',
                        choices=['gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301'], help='chatgpt model')
    parser.add_argument('--embedding-model', type=str, default='text-embedding-ada-002',
                        help='the author id of a company on twitter.')
    parser.add_argument('--data-path', type=str, default=ROOT /
                        'data/twcs.csv', help='training data path')
    parser.add_argument('--index-path', type=str, default=ROOT /
                        'data/AppleSupport_faiss_db.pkl', help='pickle file path')
    parser.add_argument('--openai-api-key', type=str,
                        help='your OPENAI api key')
    parser.add_argument('--verbose', type=bool, default=True, help='print process')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
