# add ability to segment tweets by time
# add followers tweets over time
# add followers tweets during a certain time range
import tweepy

def get_tweets(twitter_handle, user_tweets):
    '''
    Get all the tweets for a specified Twitter user
    '''
    print(f'getting tweets for {twitter_handle}')
    user_tweets[twitter_handle] = dict()
    user_tweets[twitter_handle]['tweets'] = []
    check = True
    for page in tweepy.Cursor(api.user_timeline, id=twitter_handle, tweet_mode='extended', count=200).pages():
        for tweet in page:
            if hasattr(tweet, 'retweeted_status'):
                continue        
            if check:
                user_tweets[twitter_handle]['start_date'] = tweet._json['user']['created_at']
                check = False
            user_tweets[twitter_handle]['tweets'].append((tweet.full_text, tweet.created_at))
        print(f'{len(page)} tweets found. Continuing...')
    print(f'Done getting {twitter_handle}\'s tweets!')

def get_friends(twitter_handle):
    '''
    Get all friends (Twitter users who the specified user is mutually following) and return them
    '''
    print(f'getting friends for {twitter_handle}')
    frnds = []
    for friend in tweepy.Cursor(api.friends, screen_name=twitter_handle, count=100).items():
        frnds.append(friend)
    index = 0
    while index < len(frnds):
        print(f'Checking {frnds[index]._json["screen_name"]}')
        mutual = api.show_friendship(source_screen_name=twitter_handle, target_screen_name=frnds[index]._json['screen_name'])
        if mutual[0]._json['followed_by'] == False:
            print(f'Removed {frnds[index]._json["screen_name"]}')            
            frnds.remove(frnds[index])
        else:
            index += 1
    return frnds

def sortdata(usertweets):
    output = dict()
    for name in usertweets:
        timediff = usertweets[name]['tweets'][0][1] - usertweets[name]['tweets'][-1][1]
        average_tweet = len(usertweets[name]['tweets'])/(timediff.days+1)
        active_age = timediff.days
        current_month = usertweets[name]['tweets'][0][1].month
        output[name] = dict()
        month = 0        
        output[name][month] = [usertweets[name]['tweets'][0][0]]
        for tweet in usertweets[name]['tweets']:
            if tweet[0] in output[name][0]:
                continue
            temp_m = tweet[1].month
            if temp_m == current_month:
                output[name][month].append(tweet[0])
            else:
                month += 1
                output[name][month] = [tweet[0]]
                current_month = temp_m
        for i in range(len(output[name])):
            output[name][i] = [[x] for x in output[name][i]]
    return output

