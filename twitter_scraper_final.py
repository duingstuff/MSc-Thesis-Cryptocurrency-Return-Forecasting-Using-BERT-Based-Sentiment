# Scrape Twitter Data ---------------------
# this script focuses on how to scrape Tweets data related to the cryptocurrencies

# load packages
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import time
import datetime
import copy

# scrape tweets function based on criteria such as start and end date, minimum number of favorites etc.
def scrape_tweets(limit, min_fav, min_ret, start_date, end_date):
    
    tweets_data = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('bitcoin OR btc min_faves:'+ str(min_fav) +
                                                            ' min_retweets:'+ str(min_ret) +' lang:en since:' + start_date + 
                                                            ' until:' + end_date).get_items()):
        if i>=limit:
            break
        tweets_data.append([tweet.url, tweet.date, tweet.date.year, tweet.date.month, tweet.date.day, 
                            tweet.retweetedTweet,
                            tweet.id, tweet.hashtags, tweet.content, tweet.renderedContent, 
                            tweet.likeCount, tweet.retweetCount, tweet.replyCount, 
                            tweet.user.username, tweet.user.displayname, tweet.user.description, tweet.user.verified,
                            tweet.user.favouritesCount, tweet.user.followersCount])
    # Creating a dataframe from the tweets list above
    
    #tweets_data = pd.DataFrame(tweets_data, columns=col_names)
        
    return pd.DataFrame(tweets_data)

# scrape tweets within an entire day. this is done to not exceed the "limit" for scraping
def scrape_day(limit, min_fav, min_ret, start_date, end_date):

    cont = True
    
    while (cont==True):
        tweets_data = scrape_tweets(limit, min_fav, min_ret, start_date, end_date)
        cont = len(tweets_data)<daily_limit
        if cont==True:
            min_fav = round(min_fav*0.75) # criteria
            min_ret = round(min_ret*0.75)
    
    print("Completed scraping Tweets for " + start_date + " with min_fav " + str(min_fav) + " and min_ret " + str(min_ret))
    
    return tweets_data, min_fav, min_ret

# scrape all data by iteratively scraping daily data
def scrape_all(limit, fav_limit, ret_limit, start_date, end_date):
    
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    
    col_names = ["url", "date", "year", "month", "day",
                 "retweeted_tweet",
                 "tweet ID", "hashtags", "content", "content_rendered",
                 "like_count", "retweet_count", "reply_count",
                 "username", "user_displayname", "user_description", "user_verified",
                 "user_favourites_count", "user_follower_count"]
    
    data = pd.DataFrame()
    
    cur_date_obj = copy.copy(start_date_obj)
    
    while cur_date_obj <= end_date_obj:
        cur_date = cur_date_obj.strftime('%Y-%m-%d')
    
        next_date_obj = cur_date_obj + datetime.timedelta(days=1)
        next_date = next_date_obj.strftime('%Y-%m-%d')
        
        tweets_data, actual_fav, actual_ret = scrape_day(daily_limit, fav_limit, ret_limit, cur_date, next_date)
        
        data = data.append(tweets_data)

        cur_date_obj += datetime.timedelta(days=1)    
    
    data = data.reset_index(drop=True)
    
    data.columns = col_names
    
    return data

# Creating list to append tweet data to
total_limit = None
daily_limit = 300
fav_limit = 100
ret_limit = 50

start_date = "2015-01-01"
end_date = "2017-01-01"


start_date = "2021-11-01"
end_date = "2022-02-22"


start_time = time.time()

data = scrape_all(daily_limit, fav_limit, ret_limit, start_date, end_date)

end_time = time.time()


print("Execution runtime: %s minutes" % round((end_time - start_time)/60, 2))

data.to_csv("tweets_btc_recent.csv")


# Scrape without filtering

start_date = "2019-01-01"
end_date = "2021-12-01"

start_date = "2021-10-30"
end_date = "2022-02-22"
tweets_data_e1 = []

start_time = time.time()

# entire data frame
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f"ethereum OR eth min_faves:50 min_retweets:10 lang:en since:{start_date} until:{end_date}").get_items()):
    
    print(f"Scraping for {tweet.date}")
    
    tweets_data_e1.append([tweet.url, tweet.date, tweet.date.year, tweet.date.month, tweet.date.day, 
                        tweet.retweetedTweet,
                        tweet.id, tweet.hashtags, tweet.content, tweet.renderedContent, 
                        tweet.likeCount, tweet.retweetCount, tweet.replyCount, 
                        tweet.user.username, tweet.user.displayname, tweet.user.description, tweet.user.verified,
                        tweet.user.favouritesCount, tweet.user.followersCount])
    
end_time = time.time()

print("Execution runtime: %s minutes" % round((end_time - start_time)/60, 2))

tweets_data_e1 = pd.DataFrame(tweets_data_e1)

tweets_data_e1.to_csv("tweets_data_eth_recent.csv")

tweets_data_e1.to_csv("tweets_data_eth_2019_2022.csv")

col_names = ["url", "date", "year", "month", "day",
             "retweeted_tweet",
             "tweet ID", "hashtags", "content", "content_rendered",
             "like_count", "retweet_count", "reply_count",
             "username", "user_displayname", "user_description", "user_verified",
             "user_favourites_count", "user_follower_count"]

data = pd.DataFrame()

data = data.append(tweets_data4)

data.columns = col_names

data = data.sort_values(by='date')

# save csv
data.to_csv("tweets_data_2015_2019.csv")




