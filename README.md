# Twood
IF YOU WANT TO USE THIS YOU HAVE TO GET A TWITTER API ACCESS, AND REPLACE THE KEYS AND TOKENS WITH YOUR OWN



Sentimental analysis of tweets (Twitter)

Multiple datasets that exist (Sentiment140 and the Sanders one)

Backend and frontend
Frontend:
Visualization (whatever that may be), heat graphs, based on user or hashtag and immediate connections, changes over time would be cool
Can be implemeneted in however is seen as best (website, GUI, chrome extension, whatever)
Communicate networks and relationships of sentimentality (potentially over time)

Twitter API brings in info to back end, returns requested information. The user goes into frontend and requests certain things that frontend then relays to back end. Potential things: single tweets of individual, total network of individual (various degrees of following/followers), sentiment overtime, hashtag sentiment, etc.) Maybe we can even offer different vidualization styles

Backend:
NLP analysis of tweets 
Connection to frontend

What labor needs to be done? 
Visualization (specifically within frontend)
The format of frontend, web, or whatever
Twitter API connection to "backend" (remember its not really a server database or anything, it can be in the website, I'm just talking about sending the type of information that we need to request)
Backend NLP training based off datasets

Feed to Backend
Dictionary["USERNAME-TIME SPLIT WITH ORIGIN"]= 3D LIST OF TIME TWEETS i.e. list of time separated tweets (months, days whatever), each tweet is a len=1 list that has one string with one tweet in it
