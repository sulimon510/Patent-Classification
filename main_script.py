#This is the main script for loading the data, extracting features from it,
#Dividing it into test and training data, training the model, and testing it
#The last several lines of code can be used to load a new test data set from
#a different JSON file and using it to test the model
from main_functions import *

file_name = 'uspto.json'
file_columns =['title','summary','status','filingDate','publicationDate']
data_frame=json_to_data_frame(file_name,file_columns)
train_frac=0.8
#Construct test and train sets from JSON file. 
params=EmptyClass()
params.ml_data = construct_test_and_train(data_frame,train_frac)
params.train_frac=0.8
#number of times to shuffle the data before dividing test/train sets
params.num_shuffles=100
#number of y-score thresholds to compute ROC
params.num_thresh=1000
#number of trees in random forest
params.num_estimators=20

#Fraction of features to use at each decision in random forest. The features
#used have been pre-tested for their importance by inspecting conditional 
#probability distributions, so we can use every feature.
params.feature_fraction=1.0 

#Train and test model on test and train set
mdl = train_model(params)
predictions_and_metrics=test_model(mdl,params.ml_data,params.num_thresh)
print('AUC: ' + str(predictions_and_metrics.AUC))










#Code to test model on a new file
new_file_name='uspto.json' #Using the same file, will over-fit
file_columns =['title','summary','status','filingDate','publicationDate']
data_frame=json_to_data_frame(new_file_name,file_columns)
#generate only test data from the file, i.e. 0% is training data
train_frac=0
params.ml_data = construct_test_and_train(data_frame,train_frac)
predictions_and_metrics=test_model(mdl,params.ml_data,params.num_thresh)

