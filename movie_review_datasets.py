import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from collections import Counter
from tflearn.data_utils import to_categorical

class MovieReviewDatasets:
    
    def __init__(self):

        self.review_vector_size = 10000
        self.num_categories = 2
        self.load_data()
        self.transform_data()
        
        return
    
    
    def load_data(self):
        self.df_reviews = pd.read_csv('reviews.txt', header=None)
        self.df_labels = pd.read_csv('labels.txt', header=None)
        self.num_samples = self.df_reviews.shape[0]
        
        return
    
    
    def transform_data(self):
        
        self.init_frequent_words()
        self.init_word_idx()
        self.init_review_vectors()
        self.init_sentiments()
        self.init_datasets()
        
        return
    
    
    def init_frequent_words(self):
        words = []
        for row_num, review in self.df_reviews.itertuples():
            for word in review.split(' '):
                words.append(word)

        word_count = Counter(words)
        self.frequent_words = sorted(word_count, key=word_count.get, reverse=True)[:self.review_vector_size]
        
        return
    
    
    def init_word_idx(self):
        
        self.word_idx = {}
        for idx, word in enumerate(self.frequent_words):
            self.word_idx[word] = idx
            
        return
    
    
    def init_review_vectors(self):
        
        shape = (self.num_samples, self.review_vector_size)
        self.review_vectors = np.zeros(shape)
        
        for row_num, review in self.df_reviews.itertuples():
            self.review_vectors[row_num] = self.create_review_vector(review)
            
        return
    
    
    def create_review_vector(self, review, list_unknown_words=False):
        
        review_vector = np.zeros(self.review_vector_size)
        words = review.split(' ')
        if list_unknown_words:
            unknown_words = []
        for word in words:
            try:
                idx = self.word_idx[word]
                review_vector[idx] += 1
            except:
                if list_unknown_words:
                    if not word in unknown_words:
                        unknown_words.append(word)
                next

        if list_unknown_words:
            return review_vector, unknown_words
        else:
            return review_vector
    
    
    def init_sentiments(self):
        
        self.sentiments = np.zeros(self.num_samples)
        for row_num, sentiment in self.df_labels.itertuples():
            self.sentiments[row_num] = 1.0 if (sentiment == 'positive') else 0.0
                
        return
    
    
    def init_datasets(self):

        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        
        test_fraction = 0.9
        split_index = int(self.num_samples * test_fraction)

        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        
        trainX = self.review_vectors[train_indices]
        trainY = to_categorical(self.sentiments[train_indices], self.num_categories)
        
        testX = self.review_vectors[test_indices]
        testY = to_categorical(self.sentiments[test_indices], self.num_categories)
        
        self.training_reviews = (trainX, trainY)
        self.testing_reviews = (testX, testY)
        
        return

    
    
        
