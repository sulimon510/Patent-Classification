#This code contains the main functions for training the model, training a dummy
#model, testing the model, and evaluating the model

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


#helper functions
from helper_functions import *

def train_model(params):
    #Extract parameters from params variable
    num_estimators=params.num_estimators
    feature_fraction=params.feature_fraction
    ml_data=params.ml_data
    
    #initialize RF model
    model = RandomForestClassifier(n_estimators=num_estimators,max_features=feature_fraction)
    #fit model
    model.fit(ml_data.training_features, ml_data.training_labels)

    return model
    
    
def train_dummy_model(params):
    ml_data=params.ml_data
    #initialize dummy model
    model = DummyClassifier(strategy='stratified')
    #fit model
    model.fit(ml_data.training_features, ml_data.training_labels)
    #store model
    return model
    
    
    
def test_model(model,ml_data,num_thresh):
    y_score=model.predict_proba(ml_data.test_features)
    all_metrics=calc_metrics(y_score,ml_data.test_labels,num_thresh)
    return all_metrics