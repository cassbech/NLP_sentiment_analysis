import tweepy as tw
import pandas as pd

#_______set access token for twitter API_______
consumer_key= 'zdak9yQBp5M3Vl3p3cCtHgSrL'
consumer_secret= 'QCmUE1Xxuux2IWvp8Spu0xxLiVDBK9pQ5uFpkCfAgE05LHNBTP'
access_token= '1122572932600016896-b8wjh0lWAASSvIW6v5KV6iXKRtHpAC'
access_token_secret= 'hRZGRwx2211jspyqr8DQwLHfJkIHhlju5sTjmPKXwhu3T'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

#_______scrap users tweets about Air France since 2016/01/01_______
def scrap_twitter(keywords, date_since):
""" 
scraps tweets according to keywords, through twitter API
params : the keywords to look for ; the oldest tweet date we want
returns : a dataframe includind user name, user location, tweet text and tweet date
"""
  tweets = tw.Cursor(api.search,
                    q=keywords,
                    lang="fr",
                    since=date_since,
                    tweet_mode='extended').items(2000)
  users_locs = [[tweet.user.screen_name, tweet.user.location,tweet.full_text,tweet.created_at] for tweet in tweets]
  return pd.DataFrame(data=users_locs, columns=['user', "location",'tweet','date'])

#we scrap 2 possible writings of the company : Air france / Airfrance
date_since = "2016-01-01"
keywords = "Air+France" + "-filter:retweets"
tweets_part_1=scrap_twitter(keywords, date_since)
keywords = "Airfrance" + "-filter:retweets"
tweets_part_2=scrap_twitter(keywords, date_since)
tweets_total=pd.concat([tweets_part_1,tweets_part_2])
tweets_total.drop_duplicates(inplace=True)

#remove tweets from official @AirFranceFR and prepare for supervised machine learning
tweets_total = tweets_total[tweets_total["user"]!='AirFranceFR']
tweets_total["tonalité"]=tweets_total["ironie"]=None
tweets_total.to_excel("data_tweet_sncf.xlsx", index=False)
