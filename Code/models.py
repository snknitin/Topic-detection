"""Shared inputs and interfaces for topic models."""
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import shared_utils as su

def runLDA(documents,vocab_size,numTopics):
    """

    :param documents:
    :param vocab_size:
    :param numTopics:
    :return:
    """
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=vocab_size, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Running
    print("Running Latent Dirichlet Allocation")
    t0 = time.time()
    lda = LatentDirichletAllocation(n_topics=numTopics, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)
    t1 = time.time()
    print("Seconds for LDA: %.3f"%(t1 - t0))



    # Display topics
    su.display(lda,tf_feature_names,10)

def runNMF(documents,vocab_size,numTopics):
    """
    :param documents:
    :param vocab_size:
    :param numTopics:
    :return:
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=vocab_size, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    # Running
    print("Running Non-negative Matrix Factorization")
    t0 = time.time()
    nmf = NMF(n_components=numTopics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    t1 = time.time()
    print("Seconds for NMF: %.3f"%(t1 - t0))


    # Display topics
    su.display(nmf, tfidf_feature_names, 10)