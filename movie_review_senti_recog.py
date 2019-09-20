import tflearn
import tensorflow as tf

class MovieReviewSentiRecog:
    
    
    def __init__ (self, movie_review_datasets, layer_sizes=[256, 32]):
        
        self.movie_review_datasets = movie_review_datasets
        input_size =  self.movie_review_datasets.review_vector_size
        output_size = self.movie_review_datasets.num_categories
        
        tf.reset_default_graph()
        net = tflearn.input_data([None, input_size])
        
        for layer_size in layer_sizes:
            net = tflearn.fully_connected(net, layer_size, activation='LeakyReLU')
            
        net = tflearn.fully_connected(net, output_size, activation='softmax')
        
        self.net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
        self.model = tflearn.DNN(net)
        
        return
        

    def update_training_history(self, training_state):
        self.losses     += [training_state.loss_value]
        self.accuracies += [training_state.acc_value]

        return
    
    
    def train(self, batch_size=128, n_epoch=100, exit_acc=0.9):
        
        class training_callback(tflearn.callbacks.Callback):

            def __init__(self, parent, exit_acc):
                self.parent = parent
                self.exit_acc = exit_acc
                return
            
            def on_epoch_end(self, training_state):
                self.parent.update_training_history(training_state)
                # check for early stopping
                if training_state.acc_value >= self.exit_acc:
                    # exit accuracy reached, throw error for early stopping
                    # see: https://github.com/tflearn/tflearn/issues/361
                    raise StopIteration    
                return
        
        self.losses     = []
        self.accuracies = []
        training_callback = training_callback(self, exit_acc)
        
        training_reviews, training_category_scores = self.movie_review_datasets.training_reviews
        
        try:
            self.model.fit(training_reviews, training_category_scores, callbacks=training_callback, 
                      validation_set=0.1, show_metric=True, batch_size=128, n_epoch=n_epoch)
        except:
            return
       
        return
    
    
    def get_positive_category_scores(self, sentiment_scores):
        
        return sentiment_scores[:, 1] > 0.5
    
    
    def test(self):
        
        from numpy import mean 
        
        testing_reviews, testing_category_scores = self.movie_review_datasets.testing_reviews
        prediction_category_scores = self.model.predict(testing_reviews)
        
        positive_predictions = self.get_positive_category_scores (prediction_category_scores)
        actual_positives     = self.get_positive_category_scores (testing_category_scores)
        
        accuracy = mean(positive_predictions == actual_positives)
        
        return accuracy

    
    def sentiment(self, review):
        
        r = review.lower()
        
        review_vector, self.unknown_words = self.movie_review_datasets.create_review_vector(r, list_unknown_words=True)
        
        prediction_category_scores = self.model.predict([review_vector])
        if self.get_positive_category_scores (prediction_category_scores) > 0.5:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
            
        return sentiment
