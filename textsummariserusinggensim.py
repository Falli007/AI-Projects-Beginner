#importing all the necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

#to download the stopwords
nltk.download('punkt')
nltk.download('stopwords')

#an example text to summarise
text = """
Generative AI is a type of artificial intelligence that can create new content, such as text, images, or music, based on the data it has been trained on. It uses algorithms to generate outputs that mimic human creativity. Generative AI has applications in various fields, including art, entertainment, and even scientific research. It can produce unique and original works by learning patterns and structures from existing data.
Generative AI models, such as GPT-3, have shown remarkable capabilities in understanding and generating human-like text. These models are trained on vast amounts of data and can generate coherent and contextually relevant responses. The technology is evolving rapidly, with new advancements being made regularly.    
Generative AI is also being used in industries like gaming, where it can create realistic environments and characters. In healthcare, it can assist in drug discovery by generating potential molecular structures. The potential of generative AI is vast, and it continues to push the boundaries of what machines can create.
"""

#the function to generate a summary
def generate_summary(text, num_sentences=2):
    # Tokenising the text into sentences
    sentences = sent_tokenize(text) # nltk function to split text into sentences
    words = word_tokenize(text)  # nltk function to split text into words
    
    #to filter out the stopwords and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    
    for word in words:
        if word.isalpha() and word.lower() not in stop_words:
            word_frequencies[word.lower()] = word_frequencies.get(word.lower(), 0) + 1 # Count the frequency of each word#
    
    
    #scoring sentences based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]
                    
    # Sorting sentences by their scores
    summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary = ' '.join([sentence[0] for sentence in summary_sentences[:num_sentences]])
    
    return summary

# Generating the summary
summary = generate_summary(text, num_sentences=3)
print("Original Text:\n", text) # printing the original text
print("\nSummary:\n", summary) # printing the summary

                
    