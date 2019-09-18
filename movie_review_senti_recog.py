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
        

    def update_training_history(self, training_state):
        self.losses     += [training_state.loss_value]
        self.accuracies += [training_state.acc_value]

        return
    
    
    def train(self, batch_size=128, n_epoch=100):
        
        class training_callback(tflearn.callbacks.Callback):

            def __init__(self, parent, exit_acc):
                self.parent = parent
                self.exit_acc = exit_acc
                return
            
            def on_epoch_end(self, training_state):
                self.parent.update_training_history(training_state)
                if training_state.acc_value >= self.exit_acc:
                    #--- exit accuracy reached, throw error for early stopping
                    #--- see: https://github.com/tflearn/tflearn/issues/361
                    raise StopIteration    
                return
        
        self.losses     = []
        self.accuracies = []
        training_callback = training_callback(self, exit_acc=0.99)
        
        training_reviews, training_categories = self.movie_review_datasets.training_reviews
        
        try:
            self.model.fit(training_reviews, training_categories, callbacks=training_callback, 
                      validation_set=0.1, show_metric=True, batch_size=128, n_epoch=n_epoch)
        except:
            return
       
        return
    
    
    def sentiment(self, review, list_unknown_words=True):
        
        r = review.lower()
        
        if list_unknown_words:
            review_vector, unknown_words = self.movie_review_datasets.create_review_vector(r, list_unknown_words)
        else:
            review_vector = self.movie_review_datasets.create_review_vector(r, list_unknown_words)
        
        
        prediction = self.model.predict([review_vector])
        if prediction[0][1] > 0.5:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
            
        if list_unknown_words:
            return sentiment, unknown_words
        else:
            return sentiment
            
        
        