import os

import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import data_processing as dp
import random
random.seed(42)
import numpy as np

class TopicClassifier(object):
    def __init__(self,documents,vocab_size,numTopics):
        self.numTopics=numTopics
        self.vocab_size=vocab_size
        # Shuffle samples and split in 80:20 for train and test
        random.shuffle(documents)
        self.X_train=documents[:int(np.ceil(0.8*len(documents)))]
        self.X_test=documents[int(np.ceil(0.8*len(documents))):]


    def train_classifier1(self):
        ''' Fits a classifier to the training data. '''
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=self.vocab_size, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(self.X_train)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        # Running
        tfidf_test = tfidf_vectorizer.fit_transform(self.X_test)
        tfidf_feature_names_test = tfidf_vectorizer.get_feature_names()
        self.model = NMF(n_components=self.numTopics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        self.feature_names = tfidf_feature_names
        self.W = self.model.transform(tfidf_test)


    def train_classifier2(self):
        ''' Fits a classifier to the training data. '''
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=self.vocab_size, stop_words='english')
        tf = tf_vectorizer.fit_transform(self.X_train)
        tf_feature_names = tf_vectorizer.get_feature_names()

        tf_test = tf_vectorizer.fit_transform(self.X_test)
        tf_feature_names_test = tf_vectorizer.get_feature_names()
        # Running
        print("Running Latent Dirichlet Allocation")
        t0 = time.time()
        self.model = LatentDirichletAllocation(n_topics=self.numTopics, max_iter=5, learning_method='online', learning_offset=50.,
                                        random_state=0).fit(tf)
        self.feature_names = tf_feature_names
        self.W=self.model.transform(self.X_test)

    # def predict_topics1(self):
    #     ''' Makes predictions using a fit classifier '''
    #     tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=self.vocab_size, stop_words='english')
    #     tfidf = tfidf_vectorizer.fit_transform(self.X_test)
    #     tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    #     self.model.transform(self.X_test)
    #     self.feature_names = tfidf_feature_names
    #     self.W = self.model.transform(tfidf)

    # def predict_topics2(self):
    #     ''' Makes predictions using a fit classifier '''
    #     tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=self.vocab_size, stop_words='english')
    #     tf = tf_vectorizer.fit_transform(self.X_test)
    #     tf_feature_names = tf_vectorizer.get_feature_names()
    #     self.model.transform(self.X_test)
    #     self.feature_names = tf_feature_names
    #     self.W = self.model.transform(tf)


    def display(self):
        for topic_idx, topic in enumerate(self.model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([self.feature_names[i]
                            for i in topic.argsort()[:-10 - 1:-1]]))
            top_doc_indices = np.argsort(self.W[:, topic_idx])[::-1][0:2]
            for doc_index in top_doc_indices:
                print(documents[doc_index])


if __name__=="__main__":
    documents = dp.loadDocument("E:\GIT_ROOT\\final")
    s=TopicClassifier(documents,25780,10)
    print("Running Non-negative Matrix Factorization")
    t0 = time.time()
    s.train_classifier1()
    t1 = time.time()
    print("Seconds for NMF: %.3f" % (t1 - t0))

    s.display()

    print("Running LDA")
    t0 = time.time()
    s.train_classifier2()
    t1 = time.time()
    print("Seconds for LDA: %.3f" % (t1 - t0))

    s.display()


