# this script involves combining the news, reddit and twitter datasets
# and then fitting zsc, finbert to them for future NLP models


import matplotlib.pyplot as plt
import numpy as np
from torch import nn


# news data
news = pd.read_csv("news_btc_2015-01-01_2021-12-30.csv").append(pd.read_csv("news_eth_2015-01-01_2021-12-30.csv"))
news_test = pd.read_csv("news_btc_test.csv").append(pd.read_csv("news_eth_test.csv"))[news.columns.tolist()]
news = news.append(news_test).drop(['url'],axis=1).sort_values(by=['date', 'cryptocurrency'])
news = news.drop_duplicates().rename(columns={'title':'text'})
news = news.loc[news['date'] >= "2019-01-01"].reset_index(drop=True)
news['text'] = news['text'].apply(lambda x: " - ".join(x.split(" - ")[:-1]))

news_check = news.groupby(['cryptocurrency', 'date']).size().reset_index().rename(columns={0:'count'})
news_check['count'].describe()

# tweets data
tweets = pd.read_csv("tweets_data_eth_2019_2022.csv", index_col=0).assign(cryptocurrency="ETH").append(
    pd.read_csv("tweets_data_btc_2019_2021.csv", index_col=0).assign(cryptocurrency="BTC"))
tweets_test = pd.read_csv("eth_tweets_testtttttttt.csv", index_col=0).assign(cryptocurrency="ETH").append(
    pd.read_csv("btc_tweets_testtttttttt.csv", index_col=0).assign(cryptocurrency="BTC"))
col_names = ["url", "date", "year", "month", "day",
         	"retweeted_tweet",
         	"tweet ID", "hashtags", "content", "content_rendered",
         	"like_count", "retweet_count", "reply_count",
         	"username", "user_displayname", "user_description", "user_verified",
         	"user_favourites_count", "user_follower_count", "cryptocurrency"]
tweets.columns = col_names
tweets_test.columns = col_names
tweets = tweets.append(tweets_test).sort_values(by='date').reset_index(drop=True)
tweets = tweets.loc[~tweets['date'].isna()]
tweets = tweets.loc[~tweets['username'].isna()]
tweets['date'] = pd.to_datetime(tweets['date']).dt.strftime('%Y-%m-%d')

import re
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
tweets['content'] = tweets['content'].apply(lambda x: pattern.sub('', x))

tweets_save = tweets.copy()
tweets_count = tweets_save.groupby(['cryptocurrency', 'date']).size().reset_index().rename(columns={0:'count'})

tweets = tweets_save.groupby(['cryptocurrency', 'date'], group_keys=False).apply(lambda x: x.nlargest(30,
       ['like_count', 'user_follower_count', 'retweet_count',  
        'user_favourites_count'])).reset_index(drop=True)

tweets_check = tweets.groupby(['cryptocurrency', 'date']).size().reset_index().rename(columns={0:'count'})
tweets_check['count'].describe()

tweets.groupby(['cryptocurrency', 'date']).sum().reset_index()[[
    'cryptocurrency', 'date', 'retweet_count']]

# reddit data

reddit_b = pd.read_csv("data important/reddit_bitcoin_2015_2022_filtered.csv").append(pd.read_csv("reddit_btc_test.csv"))
reddit_e = pd.read_csv("data important/reddit_ethereum_2015_2022_filtered.csv").append(pd.read_csv("reddit_eth_test.csv"))
reddit_b['cryptocurrency'] = "BTC"
reddit_e['cryptocurrency'] = "ETH"
reddit = reddit_b.append(reddit_e).sort_values(by='datetime').reset_index(drop=True)
reddit['date'] = pd.to_datetime(reddit['datetime']).dt.strftime('%Y-%m-%d')
reddit = reddit.loc[reddit['date'] >= "2019-01-01"].sort_values(by='date').reset_index(drop=True)

import re
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
reddit['title'] = reddit['title'].apply(lambda x: pattern.sub('', x))
reddit['selftext'] = reddit['selftext'].apply(lambda x: pattern.sub('', x))

reddit_save = reddit.copy()
reddit_count = reddit_save.groupby(['cryptocurrency', 'date']).size().reset_index().rename(columns={0:'count'})

reddit = reddit_save.groupby(['cryptocurrency', 'date'], group_keys=False).apply(lambda x: x.nlargest(30,
       ['num_comments', 'score'])).reset_index(drop=True)

reddit_check = reddit.groupby(['cryptocurrency', 'date']).size().reset_index().rename(columns={0:'count'})
reddit_check['count'].describe()

temp = reddit.groupby(['cryptocurrency', 'date']).sum('num_comments')

# plt.plot(temp["num_comments"], data["date"])

# combine all data

data = news.copy().assign(source="news")[['source', 'cryptocurrency', 'date', 'text']]
data = data.append(tweets.assign(source="twitter")[['source', 'cryptocurrency', 'date', 'content']].rename(
    columns={'content':'text'}))
data = data.append(reddit.assign(source="reddit")[['source', 'cryptocurrency', 'date', 'title']].rename(
    columns={'title':'text'}))
data = data.loc[~data['text'].isna()]
data = data.loc[data['date'] >= "2019-01-01"].dropna().drop_duplicates().sort_values(
    by=['cryptocurrency','date', 'source']).reset_index(drop=True)
polarity_ratios(data, ["cryptocurrency", "source"])

# data.to_excel("data_daily_100.xlsx", engine='xlsxwriter')
data = pd.read_excel("data_bert_ready.xlsx", index_col=0) # 66092 samples


# fix the date issue ----

data = pd.read_excel("data_prepared.xlsx", index_col=0) # 149654 samples

data = data.loc[data['date'] >= "2021-01-01"].drop_duplicates().reset_index(drop=True)

data['date'].loc[data['date'].apply(lambda x: len(x))>12] = pd.to_datetime(
    data['date'].loc[data['date'].apply(lambda x: len(x))>12]).dt.strftime('%Y-%m-%d')

data = data.drop_duplicates()

# data.to_excel("data_prepared_fixed.xlsx", engine='xlsxwriter')
data = pd.read_excel("data_daily_100.xlsx", index_col=0) # 66092 samples
data_pred = pd.read_excel("data_prepared_fixed.xlsx", index_col=0) # 149654 samples vs now *149609* samples

data = data.loc[data['text'].apply(lambda x: isinstance(x,str))]

data_pred = data_pred.loc[data['date'] >= "2019-08-01"].drop_duplicates().dropna().reset_index(drop=True)
data_pred = data_pred.loc[data['date'] < "2021-01-01"].drop_duplicates().dropna().reset_index(drop=True)

# data.to_excel("data_crypto_for_colab.xlsx", engine='xlsxwriter')

# bert ----

import warnings
warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = torch.load("bert_fit_unfrozen") # loading the model that I created
model.eval()

# predict using bert
def bert_fit(train, model, tokenizer):
        
    inputs, masks = encode_all(train['text'], tokenizer)
    outputs = model(inputs, token_type_ids=None, attention_mask=masks)
    logits = nn.functional.softmax(outputs[0].detach(), dim=-1).numpy()
    
    train['negative_bert'] = logits[0][0]
    train['neutral_bert'] = logits[0][1]
    train['positive_bert'] = logits[0][2]    
    train['pred_bert'] = train[['negative_bert', 'neutral_bert', 'positive_bert']].idxmax(axis=1)
    train['pred_bert'] = train['pred_bert'].apply(lambda x: x.rsplit('_bert', 1)).apply(lambda x:x[0])
    
    return train

# fit iteratively
def iterative_bert_fit(data_in, model, tokenizer):
    
    all_labeled = pd.DataFrame()
    
    for i in range(len(data_in)):
        print(i)
        tmp_labeled = bert_fit(data_in[i:(i+1)], model, tokenizer)
        
        all_labeled = all_labeled.append(tmp_labeled).reset_index(drop=True)
        
    return all_labeled


start = time.time()

bert_labeled = iterative_bert_fit(data, model, tokenizer)

end = time.time()
print("elapsed time is ", round((end - start)/60), " minutes")

data = bert_labeled.copy()

# data.to_excel("data_bert_labeled.xlsx", engine='xlsxwriter')
data = pd.read_excel("data_bert_labeled.xlsx", index_col=0) # 66126 samples



# zsc ----
# zero shot classification using BART MNLI

from transformers import pipeline
classifier = pipeline("zero-shot-classification") #bart large mnli
the_labels = ["positive", "negative"]
from sklearn.metrics import accuracy_score
from tqdm import tqdm

results = []
scores = []
for idx, item in tqdm(data.iterrows()):
    res = classifier(item['text'], the_labels, multi_label=True)
    results.append(res['labels'][0])
    scores.append(res['scores'][0])

data['label_zsc'] = results
data['score_zsc'] = scores

# data.to_excel("data_prepared_zsc_2019.xlsx", engine='xlsxwriter')
data = pd.read_excel("data_prepared_zsc_2019.xlsx", index_col=0)

data['label_zsc'].loc[data['score_zsc'] < 0.9] = "neutral"




# finbert ----
# fit finbert

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
model_name = "ProsusAI/finbert"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def model_fit_unsupervised(train, model, tokenizer):
        
    batch = tokenizer(
        train,
        padding=True,
        truncation=True,
        #max_length=512,
        max_length=32,
        return_tensors="pt",
    )
    
    outputs = model(**batch)
    
    predictions = nn.functional.softmax(outputs.logits, dim=-1)
    
    labeled = pd.DataFrame(predictions.tolist())
    labeled.columns = ["positive", "negative", "neutral"]

    labeled['text'] = train

    return labeled


def iterative_fit_unsupervised(train, model, tokenizer):
    
    all_labeled = pd.DataFrame()
    
    for idx, value in enumerate(train['text']):
        print(idx)
        tmp_labeled = model_fit_unsupervised(value, model, tokenizer)
        
        all_labeled = all_labeled.append(tmp_labeled)
        
    return all_labeled

data_finbert = iterative_fit_unsupervised(data, pt_model, tokenizer)
data_finbert['pred_finbert'] = data_finbert[['positive', 'negative', 'neutral']].idxmax(axis=1)
data = data.reset_index(drop=True)
data['positive_finbert'] = data_finbert['positive'].values
data['negative_finbert'] = data_finbert['negative'].values
data['neutral_finbert'] = data_finbert['neutral'].values
data['pred_finbert'] = data_finbert['pred_finbert'].values

# data.to_excel("data_prepared_finbert_remainder.xlsx", engine='xlsxwriter')
data = pd.read_excel("data_prepared_finbert.xlsx", index_col=0)


# entire dataset ----

data = data.merge(data_fin, on=['cryptocurrency', 'source', 'date', 'text'], how='left')

# data.to_excel("data_prepared_ready.xlsx", engine='xlsxwriter')
data = pd.read_excel("data_prepared_ready.xlsx", index_col=0)

data['label_zsc'].loc[data['score_zsc'] < 0.9] = "neutral"


# analyze and compare ----

data.groupby('pred_bert').size()

common = data.loc[(data['pred_bert'] == data['label_zsc']) & (data['pred_bert'] == data['pred_finbert'])]
common.groupby('pred_bert').size()

uncommon = data.loc[(data['pred_bert'] != data['label_zsc']) & (data['pred_bert'] != data['pred_finbert'])
                    & (data['label_zsc'] != data['pred_finbert'])]
uncommon.groupby('pred_bert').size()
uncommon.groupby('label_zsc').size()
uncommon.groupby('pred_finbert').size()


# earlier fitted (only bert) ----

# data = data.dropna().drop_duplicates().sort_values(by=['cryptocurrency', 'date', 'source']).reset_index(drop=True)
# data.to_excel("data_prepared_bert_2019.xlsx", engine='xlsxwriter')
data = pd.read_excel("data_prepared_bert_2019.xlsx", index_col=0)





