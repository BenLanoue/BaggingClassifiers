'''
By: Benjamin Lanoue
CSC 364 Machine Learning and Neural Networks
Professor Pero Atanasov

Goal: Use incrementing decision trees created by randomly
generated bootstraps to create bagging classifiers and
random forests to find the optimal amount of trees needed
to generate the best average accuracy object.

This will utilize Binary Decision Trees, 3-fold cross validation,
and plurality preprocessing of the data
'''


import math
import random
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin

from sklearn.preprocessing import Imputer

class BaggingClassifier(ClassifierMixin):
    n_trees = 8
    tree_array = None
    rf_max_features = None
    
    
    #when finished, go back to accuracy
    def __init__(self, n_trees = 8, rf_max_features = None):
        # Initialize a bagging classifier with n_trees
        # Use sklearn.tree.DecisionTreeClassifier for the decision tree implementation
        
        # Steps:
        
        # Initialize member variables to passed in values
        
        self.n_trees = n_trees
        self.rf_max_features = rf_max_features
        
        # Initialize an empty numpy array of type object to
        # hold all classifiers/trees
        self.tree_array = np.empty(self.n_trees,dtype=object)
        
        # If this is an instance of a bagging classifier (i.e. rf_max_features is None),
        # instantiate all classifiers/trees with the default constructor;
        # Otherwise, we want random forests, so instantiate them with the
        # max_features constructor
        
        if self.rf_max_features == None:
            for i in range(self.n_trees):
                self.tree_array[i] = DecisionTreeClassifier()
            #end for i
        else:
            for i in range(self.n_trees):
                self.tree_array[i] = DecisionTreeClassifier(max_features=self.rf_max_features)
        

    # end __init__
    
    def fit(self, X, y):
        # Fit the bagging classifier
        # Given a pandas dataframe of inputs X, and a pandas series y,
        # fit each of n_trees on data obtained from X and y through
        # bootstrapping, i.e., creating a training set of the same
        # size as the original by random sampling with replacement
                
        # Steps:
        
        # Put together X, y so that you can pass it to
        # __generate_bootstraps to get all your bootstraps
        # (hint pandas concat function)
        target_col_name = y.name
        df = pd.concat([X,y],axis=1)
        
        # Call __generate_bootstraps to get all your bootstraps
        
        bootstraps = self.__generate_bootstraps(df)
        

        # Use the fit method of each one of your bagging classifiers
        # to fit it on its corresponding bootstrapped data set
        
        for i in range(self.n_trees):
            temp_bt = bootstraps[i]
            
            
            temp_y = pd.Series(temp_bt[target_col_name])
            temp_y = temp_y.astype("int")
            temp_x = temp_bt.drop([target_col_name],axis = 1)
            
            self.tree_array[i].fit(temp_x,temp_y)
        #end for i
        
    # end fit
    
    
    #Pre: a Dataframe X of independent variables
    #Post: A 1d list of predictions of the Xs (as some sort of series or numpy)
    def predict(self, X):
        # Return prediction for each example in X
        # Use majority voting on the individual tree predictions
        # to determine the prediction for the bagging classifier
        
        # Steps:
        
        rows = len(X)
        
        # Initialize an empty 2-D numpy array of type object to hold
        # the predictions for each tree and for each example
        temp = np.empty(shape=(self.n_trees,rows),dtype=object)
        
        # Use the predict method of each one of your bagging classifiers
        # to get predictions for each example in X
        for i in range(self.n_trees):
            temp[i] = self.tree_array[i].predict(X)
            
        #end for i
        
        # Use pandas mode method to get the majority prediction
        # for each one of the examples and that is what should
        # be returned to the caller

        
        temp_df = pd.DataFrame(temp)
        
        return temp_df.mode(axis=0)
        

        #return temp_df.mode(axis=0)
        
    # end predict
    
    def __generate_bootstraps(self, X_y):
        # Steps:
        
        rows = len(X_y)
        columns = X_y.columns
        
        # Initialize an empty numpy array of type object to
        # hold all pandas data frames representing a 
        # bootstrapped data set for each tree
        
        bootstraps = np.empty(self.n_trees,dtype=object)

        # For each tree
        for i in range(self.n_trees):
            
            # Initialize an empty data frame for this entry
            # in your empty numpy array
            
            temp = pd.DataFrame()
            
            # For each row in original data set
            for j in range(rows):
                # Append a random row from original data set
                # into this bootstrap
                # Hints: use numpy random.randint to get a
                # random row index;
                # use pandas iloc to get that exact row from
                # original data set and then use pandas
                # append to append it into this bootstrap
                
                temp = temp.append(X_y.iloc[random.randint(0,rows-1)])
            #end for j
            
            bootstraps[i] = temp
        #end for i
        
        
        # Return each bootstrap
        return bootstraps
    
   # end __generate_bootstraps
# BaggingClassifier

#Created by Professor Pero Atanasov
def fill_missing_vals(df, missing_indicator):
    columns = df.columns.values
    for col in columns:
        col_vals = df[col].values

        for i in range(len(col_vals)):
            if col_vals[i] == missing_indicator:
                pvalue = plurality_value_mv(col_vals, col, missing_indicator)
                # Assuming here that we do not have to deal with columns that have all of their values missing
                col_vals[i] = pvalue
            # end if
        # end inner for
    # end outer for
# end fill_missing_vals

#Created by Professor Pero Atanasov
def plurality_value_mv(class_vals, class_name, missing_indicator):
    class_vals_dict = create_dict_mv(class_vals, missing_indicator)
    max_val = class_vals_dict[max(class_vals_dict, key=class_vals_dict.get)]

    max_keys = []
    for key, val in class_vals_dict.items():
        if (val == max_val):
            max_keys.append(key)
    # end for
    return random.choice(max_keys)
# end plurality_value_mv

#Created by Professor Pero Atanasov
def create_dict_mv(a, missing_indicator):
    dict = {}
    for val in a:
        if val != missing_indicator:
            if val in dict:
                dict[val] += 1
            else:
                dict[val] = 1
    return dict
# end create_dict_mv

def accuracy(cls, data):
    # Steps:
    rows = len(data)
    columns = data.columns
    num_columns = len(columns)
    
    y = pd.Series(data[columns[num_columns-1]])
    x = data.drop(columns[num_columns-1], axis = 1)
    
    
    # Get the predictions for this classifier
    # Get the actual classes for this data
    
    predictions = cls.predict(x)
    
    correct = 0
    for i in range(rows):
        
        if predictions.iloc[0][i] == y.iloc[i]:
            correct = correct+1
        #end if
    #end for
    
    return correct/rows
    
    # Calculate percentage of correct classifications
    # and return it
# end accuracy

def get_accuracy():
    # Return the accuracy of bagging and random forest classifiers
    # Use 3-fold cross-validation to return the accuracy of
    # 1. Bagging
    # 2. Random forest
    # on the data.csv or the arrhythmia.csv data set
    
    #data = pd.read_csv("data.csv", header = None)
    data = pd.read_csv("arrhythmia.csv", header=None)
    data = data.sample(frac=1)
    
    
    
    # To use the Imputer class, need to split data set into subsets
    # where all attributes will be of the same type (i.e. discrete, continuous, etc.)
    #imputer = Imputer(missing_values = "?", strategy = "most_frequent", axis = 0)
    #imputer.fit(data)
    #data = imputer.transform(data)
    
    fill_missing_vals(data, "?")
    
    sqrt_num_attrs = math.floor(math.sqrt(data.shape[1]))
    
    #data = data[:50]
    
    rows = len(data)
    columns = data.columns
    num_columns = len(columns)
    
    partition1 = int(rows/3)
    partition2 = partition1*2
    partition3 = rows
    
    
            
     #step 1: split data into 3 equal parts (randomize data beforehand)
        
    #data1,data2,data3
    dt1 = data[0:partition1]
    dt2 = data[partition1:partition2]
    dt3 = data[partition2:partition3]

  #step 2: create a validation and training set

    #training sets
    ts1 = dt2.append(dt3) #dt1 is the validation
    ts2 = dt1.append(dt3) #dt2 is the validation
    ts3 = dt1.append(dt2) #dt3 is the validation
    
    
    parameters = np.arange(1, 13, 1)
    for i in range(0, len(parameters)):

        num_trees = parameters[i]
        num_split_attrs = sqrt_num_attrs * parameters[i]
        
        # Do 3-fold cross validation for BaggingClassifier instances representing
        # (1) a bagging classifier and (2) a random forest classifier
        

        
        
      #step 3: build a bagging classifier object and .fit(training set)
        #bagging classifier objects
        
        bc1 = BaggingClassifier(num_trees,None)
        bc1rf = BaggingClassifier(num_trees,sqrt_num_attrs)
        bc2 = BaggingClassifier(num_trees,None)
        bc2rf = BaggingClassifier(num_trees,sqrt_num_attrs)
        bc3 = BaggingClassifier(num_trees,None)
        bc3rf = BaggingClassifier(num_trees,sqrt_num_attrs)
        
        
        #fitting the bagging classifiers for both bagging, and random forests
        ts1_y = pd.Series(ts1[columns[num_columns-1]])
        ts1_x = ts1.drop(columns[num_columns-1], axis = 1)
        
        ts2_y = pd.Series(ts2[columns[num_columns-1]])
        ts2_x = ts2.drop(columns[num_columns-1], axis = 1)
        
        ts3_y = pd.Series(ts3[columns[num_columns-1]])
        ts3_x = ts3.drop(columns[num_columns-1], axis = 1)
        
        bc1.fit(ts1_x,ts1_y)
        bc1rf.fit(ts1_x,ts1_y)
        
        bc2.fit(ts2_x,ts2_y)
        bc2rf.fit(ts2_x,ts2_y)
        
        bc3.fit(ts3_x,ts3_y)
        bc3rf.fit(ts3_x,ts3_y)

        
      #step 4: test for  the accuracy for the validation set and calc the average
    
        bagging_avg_accuracy = (accuracy(bc1,dt1) + accuracy(bc2,dt2) + accuracy(bc3,dt3)) / 3
        rf_avg_accuracy = (accuracy(bc1rf,dt1) + accuracy(bc2rf,dt2) + accuracy(bc3rf,dt3)) / 3
        
        
        
        

        print("BC:  num trees: {}, avg validation accuracy: {}".format(num_trees, bagging_avg_accuracy))
        print("RFC: num trees: {}, num split attrs: {}, avg validation accuracy: {}".format(num_trees, num_split_attrs, rf_avg_accuracy))
        
    # end for each parameter
# end get_accuracy

get_accuracy()
