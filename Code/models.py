"""Shared inputs and interfaces for topic models."""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


def runLDA(documents,vocab_size,numTopics):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=vocab_size, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Running
    lda = LatentDirichletAllocation(n_topics=numTopics, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)


def runNMF(documents,vocab_size,numTopics):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=vocab_size, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()