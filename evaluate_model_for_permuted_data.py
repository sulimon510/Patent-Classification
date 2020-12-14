#In this script, the model is evaluated for many versions of the test and 
#train sets. The model is generated for different random shufflings of the 
#data, and tested on the corresponding training set. The high variance shows 
#that the data set is heterogenous in predictability from the features chosen
#The parameters extracted from this analysis may be considered upper bounds
#on the best-case and worst-case possible performance of training a model
#on this data set and applying it to a data set with similar heterogeneity.

#This code also compares that result to a random "dummy" classifier
from main_functions import *
from matplotlib import pyplot

file_name = 'uspto.json'
file_columns =['title','summary','status','filingDate','publicationDate']
data_frame=json_to_data_frame(file_name,file_columns)
train_frac=0.8
#Construct test and train sets from JSON file. 
params=EmptyClass()
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

num_shufflings = 100
metrics=[0]*num_shufflings
metrics_dummy=[0]*num_shufflings
for k in range(0,num_shufflings):
    #Construct a new shuffling of the data
    params.ml_data = construct_test_and_train(data_frame,train_frac)
    mdl = train_model(params)
    #generate dummy model for comparison. dummy chooses randomly while 
    #respecting probability distribution of Granted  / not granted
    mdl_dummy=train_dummy_model(params)
    metrics[k]=test_model(mdl,params.ml_data,params.num_thresh)
    metrics_dummy[k]=test_model(mdl_dummy,params.ml_data,params.num_thresh)


