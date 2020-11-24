# need to import dependencies
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# class for Naive Bayes w/ Laplace
class NaiveBayesClassifier():
    
    def __init__(self):
        self.log_priority = {}
        self.loglikelihoods = defaultdict(defaultdict)
        self.words = []
        self.is_trained = False

    def get_words(self, training_set):
        words = set()

        for col in training_set.columns:
            words.add(col.lower())

        return words

    def count_words(self, training_set, training_labels):
        word_counts = {}
        # only two kinds of classes
        word_counts[0] = defaultdict(int)
        word_counts[1] = defaultdict(int)
        for i in range(len(training_set.index)):
            curr_row = training_set.iloc[i]
            row_words = curr_row.index[curr_row == 1]
            for word in row_words:
                word_counts[training_labels.iloc[i][0]][word] += 1
        
        return word_counts


    def train(self, training_set, training_labels, alpha=1):

        # get number of sentences in the training set
        num_sentences = len(training_set.index)

        # get a word list
        self.words = self.get_words(training_set)

        # make a set for all the different classes
        classes = set([0, 1])

        # make a dictionary for all word counts for each class
        self.word_count = self.count_words(training_set, training_labels)

        # for each class
        for c in classes:
            # get number of sentences that are of this class
            num_class_sentences = len(training_labels.loc[training_labels['CLASS'] == c].values)

            # compute log priority for the class
            self.log_priority[c] = np.log(num_class_sentences/num_sentences)

            # calculate the sum of the counts of the words in the current class
            total_count = 0
            for word in self.words:
                total_count += self.word_count[c][word]

            # compute log-likelihood & count of every word in the class
            for word in self.words:
                count = self.word_count[c][word]
                self.loglikelihoods[c][word] = np.log((count + alpha) / (total_count + alpha + len(self.words)))
        
        self.is_trained = True
    
    def predict(self, test_set, test_labels, verbose=True):
        if not self.is_trained:
            print("Classifier needs to be trained!")
            return
        
        # vars for f-measure
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        predictions = []
        prediction = -1
        for row_index, label in zip(range(len(test_set.index)), test_labels.T.values[0]):
            row = test_set.iloc[row_index]
            probabilities = self.__predictHelper(row)
            if(probabilities[0] >= probabilities[1]):
                prediction = 0
            elif (probabilities[0] < probabilities[1]):
                prediction = 1
            
            if(prediction == 1 and label == 1):
                tp += 1
                predictions.append(1)
            elif(prediction == 0 and label == 0):
                tn += 1
                predictions.append(0)
            elif(prediction == 1 and label == 0):
                fp += 1
                predictions.append(1)
            elif(prediction == 0 and label == 1):
                fn += 1
                predictions.append(0)
            
        correct = tp + tn
        total = len(test_labels.T.values[0])
        acc = round(correct/len(test_labels.T.values[0])*100,5)        

        if(verbose):
            print("Predicted {} correctly out of {} ({}%).".format(correct, total, acc))

        # f-measure calculation
        pre = tp/(tp+fp)
        rec = tp/(tp+fn)

        f_measure = (2*pre*rec)/(pre+rec)
        if(verbose):
            print("Computed f-measure: {}".format(f_measure))

        return predictions, acc, f_measure
    
    #   helper function for our main prediction function
    def __predictHelper(self, test_sentence):
        sums = {
            0: 0,
            1: 0,
        }
        classes = set([0, 1])
        for c in classes:
            sums[c] = self.log_priority[c]
            words = test_sentence.index[test_sentence == 1]
            for word in words:
                if word in self.words:
                    sums[c] += self.loglikelihoods[c][word]
        
        return sums


# make datasets
bodies_df = pd.read_csv("emails/dbworld_bodies_stemmed.csv")
subjects_df = pd.read_csv("emails/dbworld_subjects_stemmed.csv")

# need to separate them into X, y
bodies_X = bodies_df.loc[:, bodies_df.columns != "CLASS"]
bodies_X = bodies_X.loc[:, bodies_X.columns != "id"]
bodies_y = bodies_df.loc[:, bodies_df.columns == "CLASS"]

subjects_X = subjects_df.loc[:, subjects_df.columns != "CLASS"]
subjects_X = subjects_X.loc[:, subjects_X.columns != "id"]
subjects_y = subjects_df.loc[:, subjects_df.columns == "CLASS"]

# print(bodies_X)
# print(bodies_y)
# print(subjects_X)
# print(subjects_y)

# split datasets into train and test datasets
bodies_X_train, bodies_X_test, bodies_y_train, bodies_y_test = train_test_split(bodies_X, bodies_y, test_size=0.2, random_state=5)
subjects_X_train, subjects_X_test, subjects_y_train, subjects_y_test = train_test_split(subjects_X, subjects_y, test_size=0.2, random_state=5)

print("Predicting using email body dataset:")
nbc1 = NaiveBayesClassifier()
nbc1.train(bodies_X_train, bodies_y_train)
nbc1.predict(bodies_X_test, bodies_y_test)

print("\nPrediction using email subject dataset:")
nbc2 = NaiveBayesClassifier()
nbc2.train(subjects_X_train, subjects_y_train)
nbc2.predict(subjects_X_test, subjects_y_test)

def get_f_measure(predictions, labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for pred, label in zip(predictions, labels.T.values[0]):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    
    pre = tp/(tp + fp)
    rec = tp/(tp + fn)

    return (2*pre*rec)/(pre + rec)


print("\nPredicting using email body dataset and sklearn's Naive Bayes Classifier:")
nbc1 = MultinomialNB()
nbc1.fit(bodies_X_train, bodies_y_train.T.values[0])
preds = nbc1.predict(bodies_X_test)
print("f-measure: {}".format(get_f_measure(preds, bodies_y_test)))

print("\nPredicting using email subject dataset and sklearn's Naive Bayes Classifier:")
nbc2 = MultinomialNB()
nbc2.fit(subjects_X_train, subjects_y_train.T.values[0])
preds = nbc2.predict(subjects_X_test)
print("f-measure: {}\n".format(get_f_measure(preds, subjects_y_test)))