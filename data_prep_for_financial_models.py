# final preparation of the data for the financial models and trading simulation
# -------


from sklearn.preprocessing import StandardScaler

random_constant = 123
current_currency = "ETH"

# load data from the data combine step
# data = pd.read_excel("data_bert_ready.xlsx", index_col=0) # 66092 samples
data_nlp = pd.read_excel("final_data_for_evaluation_optimized.xlsx", index_col=0)
data_nlp['label'] = data_nlp['ens_4']
data_nlp = data_nlp[['source','cryptocurrency','date','text','label']]
data_nlp = data_filter(data_nlp, cur=[current_currency], source=[])  

# price data ----

if current_currency == "BTC":
    price = pd.read_csv("btc_all_price.csv", index_col=None).rename(columns={"Date":"date"})
elif current_currency == "ETH":
    price = pd.read_csv("eth_all_price.csv", index_col=None).rename(columns={"Date":"date"})

def calculate_return(df, today_or_tmr = 'tomorrow'):
    if today_or_tmr == 'tomorrow':
        return ((df.shift(-1)-df)/df).values
    if today_or_tmr == 'today':
        rtn = ((df-df.shift(1))/df.shift(1))
        rtn.bfill(inplace=True)
        return rtn.values

# price['return_today'] = np.log(price.Close) - np.log(price.Open)
# price['return_tmr'] = np.log(price['Close'].shift(-1)) - np.log(price['Close'])
price['return_daily'] = calculate_return(price['Close'], 'today')
price['return_tmr'] = calculate_return(price['Close'], 'tomorrow')
price['Close_tmr'] = price['Close'].shift(-1)
price['dif_tmr'] = price['Close'].shift(-1) - price['Close']
price['rel_p_change'] = 2*(price['High']-price['Low'])/(price['High']+price['Low'])
price['parkinson_volatility'] = np.sqrt( np.log(price['High']/price['Low'])**2 / (4*np.log(2)) )

polarity_ratios(data_nlp, ["cryptocurrency", "source"])
polarity_ratios(data_nlp, ["label"])

# function to filter data based on sentiment label
def filter_label(df, label):
    
    df = df.loc[df.label==label].reset_index(drop=True).rename(columns={0:"count"})
    
    return df

# create a table of all polarities gathered in a daily granularity
def create_polarity_table(df, currency):
    
    pol = df.groupby(['cryptocurrency', 'source', 'date', 'label']).size().reset_index()
    pol['sentiment'] = pol['source']+"_"+pol['label']
    pol = data_filter(pol, cur=[currency])
    pol = pol.drop(['source', 'label'], axis=1).rename(columns={0:"count"})
    
    # pd.melt(pol, id_vars="count", value_vars = ["sentiment", 'date'])
    pol_spread = pol.set_index(['cryptocurrency','date']).pivot(columns='sentiment').reset_index().fillna(0)
    pol_spread.columns = [col[0] if col[1]=='' else col[1] for col in pol_spread.columns.values]
    # pol_spread.columns = ['_'.join(col) for col in pol_spread.columns.values]
    
    pol_spread['news_count'] = pol_spread.filter(like='news').sum(axis=1).values
    pol_spread['reddit_count'] = pol_spread.filter(like='reddit').sum(axis=1).values
    pol_spread['twitter_count'] = pol_spread.filter(like='twitter').sum(axis=1).values
    
    return pol_spread

# aggregate sentiment
def get_each_sentiment(pol_spread):
    
    sent = pd.DataFrame()
    sent['sentiment_news'] = (pol_spread['news_positive'] - pol_spread['news_negative']) / pol_spread['news_count'] 
    sent['sentiment_twitter'] = (pol_spread['twitter_positive'] - pol_spread['twitter_negative']) / pol_spread['twitter_count']
    sent['sentiment_reddit'] = (pol_spread['reddit_positive'] - pol_spread['reddit_negative']) / pol_spread['reddit_count']
        
    sent = sent.fillna(0)
    
    return sent

# create sentiment features by weighted averaging
def create_sentiment_features(pol_spread, price_data, weights=[1/3,1/3,1/3]):
    
    # sentiment_n = pol_spread['news_positive'] / (pol_spread['news_positive'] + 
    #                                              pol_spread['news_negative']) - 0.5
    # sentiment_t = pol_spread['twitter_positive'] / (pol_spread['twitter_positive'] + 
    #                                              pol_spread['twitter_negative']) - 0.5
    # sentiment_r = pol_spread['reddit_positive'] / (pol_spread['reddit_positive'] + 
    #                                              pol_spread['reddit_negative']) - 0.5

    # sentiment_n = (pol_spread['news_positive'] - pol_spread['news_negative'])
    # sentiment_t = (pol_spread['twitter_positive'] - pol_spread['twitter_negative'])
    # sentiment_r = (pol_spread['reddit_positive'] - pol_spread['reddit_negative'])
    
    sentiment_df = get_each_sentiment(pol_spread)
        
    # scale = StandardScaler()
    # sent_standard = scale.fit_transform(sentiment)
    # sentiment = pd.DataFrame(sent_standard, columns=['n','t','r'])
    
    sentiment_overall = (weights[0]*sentiment_df.sentiment_news.values + 
                         weights[1]*sentiment_df.sentiment_twitter.values + 
                                    weights[2]*sentiment_df.sentiment_reddit.values)
    
    sent_feat = pol_spread[['cryptocurrency', 'date']].assign(sentiment_overall = sentiment_overall)
    sent_feat.columns = [''.join(col) for col in sent_feat.columns.values]
    
    df = price.merge(sent_feat, on='date', how='right')

    return df

# create nlp dataset
train_nlp = data_filter(data_nlp, cur=[], source=[], date_min="", date_max="2021-10-31")
polarity_table_train = create_polarity_table(train_nlp, current_currency)

# aggregate sentiment features using input weight
sentiment = create_sentiment_features(polarity_table_train, price, [1/3,1/3,1/3])
sentiment['sentiment_overall'].describe()
sentiment_corr(sentiment, target)

plt.plot(sentiment['sentiment_overall'])

# compute correlation of sentiment with the target
def sentiment_corr(df, target):
    
    return df[[target, 'sentiment_overall']].corr().sentiment_overall[0]

# compute granger causality matrix to see sentiment versus target
def grangers_causation_matrix(data, variables, maxlag, test='ssr_chi2test', verbose=False):    
   
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

# compute granger causality of target
def granger_sentiment_target(df, target, sent_feat, maxlag, test='ssr_chi2test'):
    granger_matrix = grangers_causation_matrix(df, variables=[target, sent_feat], maxlag=maxlag)
    sentiment_target_causality = granger_matrix[sent_feat+'_x'][target+'_y']
    return sentiment_target_causality

# find optimal weights of the 3 sentiments based on causality measure
# UPDATE: THIS IS NOT USED! WE TAKE AN UNWEIGHTED APPROACH IN THE END....
def find_sentiment_weights(pol, price_data, target, granger_max_lag=15):
    
    w_values = np.arange(0.0, 1.1, 0.05).round(2)
    w_optimal = []
    # sent_corr_max = 0
    sent_granger_min = 1
    output = pd.DataFrame(columns=['w_n','w_t','w_r','granger_causality'])
    
    for w_n in w_values:
        for w_t in w_values:
            for w_r in w_values:
                if (w_n+w_t+w_r)==1.0:
                    sent_temp = create_sentiment_features(pol, price_data, weights=[w_n, w_t, w_r])
                    # sent_corr = sentiment_corr(sent_temp, target)
                    # print("Correlation with ", str([w_n, w_t, w_r]), " is ", sent_corr.round(4))
                    sent_granger = granger_sentiment_target(sent_temp, 
                                                            target, 'sentiment_overall', maxlag=granger_max_lag)
                    print("Granger Causality with ", str([w_n, w_t, w_r]), " is ", sent_granger.round(4))
                    output = output.append(pd.DataFrame({"w_n":[w_n],"w_t":[w_t],"w_r":[w_r],
                                                         "granger_causality":[sent_granger]}))
                    # if abs(sent_corr) > abs(sent_corr_max):
                    if sent_granger < sent_granger_min:
                        w_optimal = [w_n, w_t, w_r]
                        # sent_corr_max = sent_corr
                        sent_granger_min = sent_granger
    print("--Optimal sentiment weights: ", w_optimal, "\n--Granger Causality with target: ", sent_granger_min)
    return w_optimal, sent_granger_min, output.sort_values('granger_causality', ascending=True)

# select target (uncomment that one)
target = 'Close_tmr'
target = 'return_tmr'
target = 'dif_tmr'

# find optimal weights that maximize corr with target in train dataset ----
w_optimal, corr_max, granger_compare_table = find_sentiment_weights(polarity_table_train.copy(), price, target, 15)
granger_compare_table.head(10) # select sentiment feature from lowest granger causality scores

# apply the optimal weights to entire dataset ----
polarity_table = create_polarity_table(data_nlp, current_currency)
data_source_sentiment = get_each_sentiment(polarity_table.copy())
sentiment = create_sentiment_features(polarity_table, price, w_optimal)
sentiment = pd.concat([sentiment, data_source_sentiment], axis=1)
sentiment_corr(sentiment, target)
sentiment['sentiment_overall'].describe()

# plot sentiment
plt.plot(sentiment['sentiment_overall'])
plt.plot((sentiment['Close_tmr'] - sentiment['Close_tmr'].mean())/sentiment['Close_tmr'].mean()*0.2)

# corr plot
corr = sentiment.drop(['date'],axis=1).corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()

sentiment[['return_tmr','sentiment_overall']].corr()
sentiment['sentiment_overall'].describe()




# -------------------------------------------------------------------------------------------------------------



# final price and sentiment dataset ----

data = sentiment.copy()
data = data_filter(data, cur=[], source=[], date_min="2019-08-01", date_max="2022-02-15")

# split as train and test datasets ----

train = data_filter(data, cur=[], source=[], date_min="2019-08-01", date_max="2021-10-31")
test = data_filter(data, cur=[], source=[], date_min="2021-11-01", date_max="2022-02-15")

# ---- add financial features ----

# daily post count feature ----

data_count = pd.read_excel("data_daily_100.xlsx", index_col=0)

data_count = data_count.groupby(['cryptocurrency','source','date']).size().reset_index().rename(
                                                        columns={0:'count'})


data_count = data_count.pivot(index=['cryptocurrency','date'], columns='source').reset_index().fillna(0)
data_count.columns = ['cryptocurrency','date','count_news','count_reddit','count_twitter']

data = data.merge(data_count, on=['cryptocurrency','date'], how='left')
data['count_total'] = data['count_news'] + data['count_twitter'] + data['count_reddit']

plt.plot(data.count_total)
plt.plot((data['Close'] - data['Close'].mean())/data['Close'].mean()*50+100)

plt.plot(data.sentiment_overall * data.count_total)
plt.plot((data['Close'] - data['Close'].mean())/data['Close'].mean()*20)


# S&P 500 feature ----

import pandas_datareader.data as web

# scrape the S&P 500 data
SP500 = web.DataReader(['sp500'], 'fred', start=data.date.min(), end=data.date.max())

plot_data = pd.DataFrame()
plot_data['btc_price'] = data['Close'].values
plot_data.index = data.date.values
plot_data_sp500 = pd.DataFrame()
plot_data_sp500['sp500'] = SP500.sp500.values
plot_data_sp500.index = SP500.index.strftime('%Y-%m-%d').values
plot_data = plot_data.join(plot_data_sp500, how='left')
plot_data['date'] = plot_data.index

#plot_data.to_excel("sp500_data.xlsx", engine='xlsxwriter')
# plot_data = pd.read_excel("sp500_data.xlsx", index_col=0)

# remove nan by interpolation
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

nans, x= nan_helper(plot_data.sp500.values)
plot_data['sp500'][nans] = np.interp(x(nans), x(~nans), plot_data['sp500'][~nans])

plt.figure(figsize=(12,5))
ax1 = plot_data.sp500.plot(color='blue', grid=True, label='S&P 500')
ax2 = plot_data.btc_price.plot(color='red', secondary_y=True, grid=True, label='BTC/USD')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1+h2, l1+l2, loc=2)
plt.show()

data = data.merge(plot_data.drop(['btc_price'], axis=1).reset_index(drop=True), how='left')



# compute market volatility using S&P 500 and currency volatility using Closing price ----

data['volatility_index'] = data.sp500.rolling(window=30).std().values
data.volatility_index.bfill(inplace=True)

data['currency_volatility'] = data.Close.rolling(window=30).std().values
data.currency_volatility.bfill(inplace=True)

# plot volatility
data['volatility_index'].plot(style='b')

plt.figure(figsize=(12,5))
ax1 = data.volatility_index.plot(color='blue', grid=True, label='Volatility Index')
ax2 = data.Close.plot(color='red', secondary_y=True, grid=True, label='BTC/USD')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1+h2, l1+l2, loc=2)
plt.show()



# stock index returns ----

data['return_sp500'] = calculate_return(data['sp500'], 'today')

plt.figure(figsize=(12,5))
ax1 = data.return_sp500.plot(color='blue', grid=True, label='S&P 500 Return')
ax2 = data.return_tmr.plot(color='red', secondary_y=True, grid=True, label='Crypto Closing Price Return')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1+h2, l1+l2, loc=2)
plt.show()


#data.to_excel("data_sofarsogood.xlsx", engine='xlsxwriter')
data = pd.read_excel("data_sofarsogood.xlsx", index_col=0)

# inflation and market data ----

inf = pd.read_excel("findata/inf_rate_breakeven.xls", index_col=None)
markt =  pd.read_excel("findata/market_rate_treasury.xls", index_col=None)
inf['date'] = pd.to_datetime(inf['date']).dt.strftime('%Y-%m-%d')
markt['date'] = pd.to_datetime(markt['date']).dt.strftime('%Y-%m-%d')

data = data.merge(inf, on='date', how='left')
data = data.merge(markt, on='date', how='left')


# currency market cap ----
# only for BTC

if current_currency == "BTC":
    cap = pd.read_excel("findata/marketcapitalization_coinmarketcap_btc.xlsx", index_col=None)
    cap = cap[['Date', 'Market Cap']].rename(columns={'Date':'date', 'Market Cap':'market_cap'})
    cap['date'] = pd.to_datetime(cap['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(cap, on='date', how='left')


# blockchain data ----
# only for BTC

if current_currency == "BTC":
    
    n_trans = pd.read_csv("findata/blockchain data btc/n-transactions ma7")
    n_trans = n_trans.rename(columns={'Timestamp':'date', 'n-transactions':'n_transac_ma7'})
    n_trans['date'] = pd.to_datetime(n_trans['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(n_trans, on='date', how='left')
    
    n_trans = pd.read_csv("findata/blockchain data btc/n-transactions")
    n_trans = n_trans.rename(columns={'Timestamp':'date', 'n-transactions':'n_transac'})
    n_trans['date'] = pd.to_datetime(n_trans['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(n_trans, on='date', how='left')
    
    hash_r = pd.read_csv("findata/blockchain data btc/total hash rate ma7")
    hash_r = hash_r.rename(columns={'Timestamp':'date', 'hash-rate':'hash_rate_ma7'})
    hash_r['date'] = pd.to_datetime(hash_r['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(hash_r, on='date', how='left')
    
    net_dif = pd.read_csv("findata/blockchain data btc/network difficulty")
    net_dif = net_dif.rename(columns={'Timestamp':'date', 'difficulty':'network_difficulty'})
    net_dif['date'] = pd.to_datetime(net_dif['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(net_dif, on='date', how='left')
    
    miner_rev = pd.read_csv("findata/blockchain data btc/miners revenue")
    miner_rev = miner_rev.rename(columns={'Timestamp':'date', 'miners-revenue':'miners_rev'})
    miner_rev['date'] = pd.to_datetime(miner_rev['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(miner_rev, on='date', how='left')
    
    av_block = pd.read_csv("findata/blockchain data btc/average block size ma7")
    av_block = av_block.rename(columns={'Timestamp':'date', 'avg-block-size':'avg_block_size_ma7'})
    av_block['date'] = pd.to_datetime(av_block['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(av_block, on='date', how='left')
    
    transac_vol = pd.read_csv("findata/blockchain data btc/estimated-transaction-volume")
    transac_vol = transac_vol.rename(columns={'Timestamp':'date', 
                                              'estimated-transaction-volume':'est_transac_volume'})
    transac_vol['date'] = pd.to_datetime(transac_vol['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(transac_vol, on='date', how='left')
    
    av_trans_block = pd.read_csv("findata/blockchain data btc/average transactions per block ma7")
    av_trans_block = av_trans_block.rename(columns={'Timestamp':'date', 
                                                    'n-transactions-per-block':'transac_per_block'})
    av_trans_block['date'] = pd.to_datetime(av_trans_block['date']).dt.strftime('%Y-%m-%d')
    data = data.merge(av_trans_block, on='date', how='left')

# weekday dummies

weekday_info = pd.to_datetime(data.date.values).strftime('%A').tolist()

dummies = pd.get_dummies(weekday_info).rename(columns=lambda x: 'weekday_' + str(x))

data = pd.concat([data, dummies], axis=1)


# add the price of other currency ----

if current_currency == "ETH":
    price_other = pd.read_csv("btc_all_price.csv", index_col=None).rename(columns={"Date":"date"})
elif current_currency == "BTC":
    price_other = pd.read_csv("eth_all_price.csv", index_col=None).rename(columns={"Date":"date"})

data = data.merge(price_other[['date', 'Close', 'Volume']].rename(columns={'Close':"price_other", 
                                         'Volume':"volume_other"}), on='date', how='left')

data['return_daily_other'] = calculate_return(data['price_other'], 'today')


# linear interpolation of missing values ----

nans, x= nan_helper(plot_data.sp500.values)
plot_data['sp500'][nans] = np.interp(x(nans), x(~nans), plot_data['sp500'][~nans])

def interpolate_nans(df, all_feat):
    
    for feat in all_feat:
        nans, x = nan_helper(df[feat].values)
        df[feat][nans] = np.interp(x(nans), x(~nans), df[feat][~nans])
    
    return df

feat_interpolate = data.columns[data.isnull().sum()!=0]

data = interpolate_nans(data.copy(), feat_interpolate)


