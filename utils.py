import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def data_preprocessing(csv_file, company: str) -> pd.DataFrame:
    tqdm.pandas(desc='Preprocessing data')

    df = pd.read_csv(csv_file, encoding='utf-8')

    # Drop duplicated tweet_id row
    df = df.drop_duplicates(subset='tweet_id', keep='first')

    # Fill missing values with -1
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna(
        -1).astype(int)
    df['response_tweet_id'] = df['response_tweet_id'].fillna(-1)

    # Add a new column for the associated company
    # Label response from customer support
    df["tweet_type"] = "None"
    df.loc[(df["author_id"] == company) & (
        df["inbound"] == False), "tweet_type"] = "response"

    # Label questions from users

    # If word after'@' match the selected company
    extracted_target = df['text'].str.extract(r'@(\w+)', expand=False)
    question_tweet_id_list = df[extracted_target.str.match(
        company) == True]['tweet_id'].astype(int).tolist()

    # Get in_response_to_tweet_id and respond_tweet_id
    company_df = df[df["tweet_type"] == "response"]
    question_tweet_id_list += company_df.loc[:,
                                             "in_response_to_tweet_id"].astype(int).tolist()
    question_tweet_id_list += company_df.loc[:, "response_tweet_id"].str.split(
        ',').fillna(-1).explode().astype(int).tolist()

    question_tweet_id_list = list(set(question_tweet_id_list))

    df.loc[df['tweet_id'].isin(question_tweet_id_list),
           'tweet_type'] = 'question'
    df.loc[df['tweet_type'] == 'None', 'tweet_type'] = np.nan

    df = df[df['tweet_type'].notna()]

    return df


def get_texts(df) -> list:

    threads_text = []  # conversation threads text

    for i, row in tqdm(df.iterrows(), total=df.shape[0], mininterval=0, desc='Converting to text'):

        tweet_id = row['tweet_id']
        conversation = []

        while tweet_id:
            row = df.loc[df["tweet_id"] == tweet_id]
            if not row.empty:
                row = row.squeeze()

                info = (pd.to_datetime(row['created_at']),
                        row['tweet_id'], row['inbound'])
                conversation.append(info)

                response_tweet_id = row["response_tweet_id"]
                in_response_to_tweet_id = row["in_response_to_tweet_id"]

                if pd.notna(response_tweet_id):
                    tweet_id = int(str(response_tweet_id).split(',')[0])
                elif pd.notna(in_response_to_tweet_id):
                    tweet_id = int(str(in_response_to_tweet_id).split(',')[0])
                else:
                    tweet_id = None
            else:
                tweet_id = None

        conversation = sorted(conversation, key=lambda x: x[0])

        if len(conversation) == 0:
            continue

        # Convert into expected format
        current_thread_text = ""

        for _, chain_id, inbound in conversation:
            text = df.loc[df['tweet_id'] == chain_id, 'text'].values[0]

            if inbound:
                new_text = f"User:\n{text}\n"
            else:
                new_text = f"Support:\n{text}\n"

            current_thread_text += (new_text)

        threads_text.append(current_thread_text)

    return threads_text
