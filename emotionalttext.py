import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import pandas as pd

#To download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

#initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

#Sample text data for the emotional text analysis
text_data = """
ILondon is a vibrant city with a rich history and diverse culture. The streets are filled with people from all walks of life, each contributing to the unique tapestry of the city. From the iconic red buses to the historic landmarks, London offers a blend of tradition and modernity that captivates both residents and visitors alike.
The Thames River flows through the heart of the city, providing a picturesque backdrop for leisurely strolls and boat rides. The parks are lush and green, offering a peaceful escape from the bustling urban environment. The sound of laughter and chatter fills the air as people gather in cafes and restaurants, enjoying the culinary delights that London has to offer.
The arts scene is thriving, with galleries, theaters, and music venues showcasing a wide range of talent. From classical performances to contemporary art exhibitions, there is something for everyone to enjoy. The city's multiculturalism is reflected in its festivals and events, celebrating the rich tapestry of cultures that make up London.
The people of London are known for their resilience and adaptability, having weathered numerous challenges throughout history. The spirit of innovation and creativity is palpable, with startups and established businesses alike pushing the boundaries of what is possible.
In conclusion, London is a city that never fails to inspire and amaze. Its blend of history, culture, and modernity creates a unique atmosphere that is both exciting and comforting. Whether you're exploring the historic streets or enjoying the vibrant nightlife, London has something to offer everyone.
"""

#Function to preprocess the text
def detect_emotion(text):
    #analyze the sentiment of the text
    scores = sid.polarity_scores(text)
    
    #diasplay the sentiment scores
    print("Sentiment Scores:", scores)
    
    #determine the overall sentiment
    if scores['compound'] >= 0.05:
        emotion = "Joy"
    elif scores['compound'] <= -0.05:
        emotion = "Sadness"
    elif scores['neg'] > 0.05:
        emotion = "Anger"
    elif scores['neu'] > 0.05:
        emotion = "Neutral"
    else:
        emotion = "Mixed Emotions"

#display the detected emotion
emotion = detect_emotion(text_data)
print("Detected Emotion:", emotion)
