
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from keras import models
import os
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
        if (len(usertweets[name]['tweets']) == 0):
            continue
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
    
if __name__ == '__main__':
    consumer_key = os.environ.get('key')
    consumer_secret = os.environ.get('secret')
    access_token = os.environ.get('token')
    access_token_secret = os.environ.get('token_secret')
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
#    while True:
#        option = int(input('Would you like to search for your Twitter network or just yourself? (1, 2) ').strip())
#        if option != 1 and option != 2:
#            print(f'{option} is not one of the valid responses. Just press 1 or 2')
#        else:
#            break
#    twitter_handle = input('What is the Twitter handle of the user you\'d like to search? ').strip()
#    user_tweets = dict()
#    get_tweets(twitter_handle, user_tweets)
#    if option == 1:
#        friends = get_friends(twitter_handle)
#        for fren in friends:
#            if fren._json['protected']:
#                continue
#            get_tweets(fren._json['screen_name'], user_tweets)
#    
#    user_tweets = sortdata(user_tweets)
#    print(user_tweets)

#----------------------------------------------------------------------------------------------------------------------------------

def singleUser(data,time,user):
    print(data)
    plt.plot(time,data,'c')
    plt.title("Mood of User vs Time"+user)
    plt.ylabel("Mood Scale")
    plt.xlabel("Time (months)")
    plt.show()

def userFollowers(username, data, time):
    #print(data)
    g=nx.DiGraph()
    g.add_node(username,stuff=data[username][time])
    nodeColor=[data[username][time]]
    sizeArray=[1200]
    dict={username:(0,0)}
    labeldict={}
    j=1
    
    for x in data:
        if(x==username): continue
        g.add_node(x,stuff=data[x][time])
        tWeight=abs(g.nodes[username]['stuff']-data[x][time])
        g.add_edge(username,x,weight=tWeight)
        nodeColor.append(data[x][time])
        sizeArray.append(300)
        angle=2*math.pi*(j/len(data))
        xfac=math.cos(angle)/abs(math.cos(angle))*1000*abs(math.cos(angle))
        yfac=math.sin(angle)/abs(math.sin(angle))*1000*abs(math.sin(angle))
        dict[x]=(xfac+(tWeight/4)*4000*math.cos(angle),yfac+(tWeight/4)*4000*math.sin(angle))
        labeldict[x]=x
        j+=1
    
    nodes=nx.draw_networkx_nodes(g,pos=dict,node_color=nodeColor,cmap=plt.cm.plasma,node_size=sizeArray)
    edgeColor=nodeColor
    #print(edgeColor)
    edgeColor.pop(0)
    #print(edgeColor)
    nx.draw_networkx_edges(g,pos=dict,edge_color=edgeColor,edge_cmap=plt.cm.plasma,width=2)
    nx.draw_networkx_labels(g,pos=dict,labels={username:username},font_color='c')
    nx.draw_networkx_labels(g,pos=dict,labels=labeldict,font_color='c',font_size=8)
    # make the plot
    plt.axis('off')
    plt.plot()
    plt.title("Graph Relations of Friends and their Relative Moods")
    pc=mpl.collections.PathCollection(nodes,cmap=plt.cm.plasma)
    pc.set_array(nodeColor)
    plt.colorbar(pc)

#datadata=[]
#time=[]
#user="User"
#timescale="days"
#howMany=5
#for y in range(0,howMany):
#    data={}
#    for x in range(0,100):
#        data[random_generator(9)]=(4.0*ran.random())
#        time.append(x)
#    data[user]=(4.0*ran.random())
#    datadata.append(data)
#random data for testing
s=input("Twitter handle (no punctuation) and analysis style (self or group)")
args=s.split()

model = load_model("finalmodel.h5")
with open('finaltoken.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

if args[1]=="self":
    #get data  stuff
    data=dict()
    get_tweets(args[0],data)
    sorted=sortdata(data)
    sentences = []
    num_time = num_tweets = 0
    for i, j in sorted.items():
        num_time = max(num_time, len(j))
        for m, n in j.items():
            num_tweets = max(num_tweets, len(n))
    results = np.zeros(shape=(len(sorted), num_time, num_tweets), dtype=float)
    user = 0
    for i, j in sorted.items():
        for m, n in j.items():
            sentences.clear()
            for p in n:
                sentences.append(p[0])
            #print(sentences)
            seqeunces = tokenizer.texts_to_sequences(sentences)
            data = pad_sequences(seqeunces, padding='post', maxlen=40)
            #print(data)
            out = model.predict(data)
            for q in range(len(out)):
                results[user][m][q] = out[q][0]
            #print(model.predict(data))
        user += 1
    #print(results[0].shape)
    #print(results[0])
    final = []
    for i in results[0]:
        summ = 0
        num = 0
        for j in i:
            if (j != 0):
                summ += j
                num += 1
        if (num != 0):
            avg = summ/num
        else:
            avg = 0
        final.append(avg)
    
    #print(final)

    singleUser(final, np.arange(len(final)), args[0])
elif args[1]=="group":
    #get data
    data=dict()
    get_tweets(args[0],data)
    frens=get_friends(args[0])
    for fren in frens:
        get_tweets(fren._json["screen_name"],data)
    sorted=sortdata(data)
    sentences = []
    num_time = num_tweets = 0
    for i, j in sorted.items():
        num_time = max(num_time, len(j))
        for m, n in j.items():
            num_tweets = max(num_tweets, len(n))
    results = np.zeros(shape=(len(sorted), num_time, num_tweets), dtype=float)
    user = 0
    users = []
    for i, j in sorted.items():
        for m, n in j.items():
            sentences.clear()
            for p in n:
                sentences.append(p[0])
            #print(sentences)
            seqeunces = tokenizer.texts_to_sequences(sentences)
            data = pad_sequences(seqeunces, padding='post', maxlen=40)
            #print(data)
            out = model.predict(data)
            for q in range(len(out)):
                results[user][m][q] = out[q][0]
            #print(model.predict(data))
        users.append(i)
        user += 1
    #print(results[0].shape)
    #print(results[0])
    final = dict()
    avgs = []
    p = 0
    #print(results)
    #print(users)
    for i in results:
        avgs.clear()
        for j in i:
            summ = 0
            num = 0
            for k in j:
                if (k != 0):
                    summ += k
                    num += 1
            if (num != 0):
                avg = summ/num
            else:
                avg = 0
            avgs.append(avg)
            #print(avgs)
        final[users[p]] = avgs.copy()
        #print(final)
        p += 1
    #print(final)
    while True:
        s=input("timeline from 0 to "+ str(num_time-1) +" (-1 to exit)\n")
        if int(s)==-1 :break
        plt.figure(int(s))
        userFollowers(users[0],final,int(s))
        plt.show()
else:
    print("Invalid Arguements!")
    exit(1)
