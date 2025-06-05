import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#sample resumes and job descriptions data 

data = {
    'resume_id': [1, 2, 3],
    'resume_text': [
        "Experienced software engineer with expertise in Python and machine learning.",
        "Data scientist with a strong background in statistics and data analysis.",
        "Project manager with experience in agile methodologies and team leadership."
    ]
}

job_descriptions = "Software engineer with skills in Python, machine learning, and agile development."

#convert the data into a DataFrame
df = pd.DataFrame(data)
print("Resumes:\n", df)

#Combine job description and resumes for vectorization
documents = df['resume_text'].tolist() + [job_descriptions]

#Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

#Calculate similarity scores between job description and resumes
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()


#display the similarity scores for each resume 
df['similarity_score'] = similarity_scores
print("\nSimilarity Scores:\n", df[['resume_id', 'similarity_score']])

#Identify resumes that match the job description (threshold can be adjusted)
threshold = 0.2
matched_resumes = df[df['similarity_score'] >= threshold]
print("\nResumes matching the job requirement:\n", matched_resumes[['resume_id', 'similarity_score']])