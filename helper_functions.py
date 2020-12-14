#This file contains all helper functions for loading the data, cleaning the 
#data, choosing features, dividing the data into test and train sets, and
#calculating evaluation metrics

import json    
import pandas as pd
from datetime import datetime
import numpy as np

#empty class to later fill in test and training data and pass around
class EmptyClass:
    pass

#import dataframe and fill into pandas dataframe
def json_to_data_frame(filename, cols):
    #create an empty data frame with the given columns
    data_frame = pd.DataFrame(columns=cols)

    with open('uspto.json') as f:
        for line in f:
            
            curr_line=json.loads(line);
            curr_row_data=[]
            #create a list of lists, containing data from new column 
            for col in cols:
                curr_row_data.append(curr_line['object'][col])
            
 
            extracted_data = pd.DataFrame([curr_row_data],columns=cols)
            data_frame=data_frame.append(extracted_data)

    return data_frame
    
#extract relevant features from data-frame and convert features to numeric
#format for ML classification
def extract_features(data_frame):

    #extract title lengths, summary lengths
    titles=data_frame['title']
    summaries=data_frame['summary']
    title_len=titles.str.len()
    summary_len=summaries.str.len()
    
    #extract whether title is all upper-case
    title_case= [mystring.isupper() for i, mystring in enumerate(titles)]
    
    #extract filing date and publish date in the string format given 
    pub_date_str = list(data_frame['publicationDate'])
    file_date_str=list(data_frame['filingDate'])
    
    #Convert the publishing and filing dates to a decimal number representing 
    #years. 
    pub_date_numeric = get_dates(pub_date_str)
    file_time_numeric = get_dates(file_date_str)
    
    
    #set up training set from the above data
    
    total_features = np.asarray([title_len,summary_len,file_time_numeric,pub_date_numeric,title_case])
    total_features = np.transpose(total_features)
    return total_features;
  
#helper function to convert date strings to numeric values
def get_dates(date_strings):
    #Convert dates to a decimal number representing years. For example one day 
    #counts as 1/365 years, so January 2nd 2016 is roughly equal to 2016.0027
    #Replace missing years, (meaning never filed or publsihed) with 2019
        date_numeric = []
        for i in range(0,len(date_strings)):
            if date_strings[i]!='': #Date is not empty
                cleaned_string = date_strings[i][4:10]+date_strings[i][25:30]
                datetime_obj = datetime.strptime(cleaned_string,'%b %d %Y')
                date_numeric.append(float(datetime_obj.year)+float(datetime_obj.month/12.0)+float(datetime_obj.day/365.0))
            else: #date is empty, meaning hasn't occured yet
                date_numeric.append(2020)
        return date_numeric


#take the data frame, convert to numpy array, and split into test/train sets
def construct_test_and_train(data_frame,train_frac):

    #Randomly shuffle the data
    data_frame=data_frame.sample(frac=1)
    num_train = int(len(data_frame)*train_frac)
    num_total=len(data_frame)
    #Call the helper function which converts features dataframe to np array
    total_features=extract_features(data_frame)
    total_labels=np.transpose(np.array(data_frame['status']=='Patented Case'))

    #set up test and training features and labels
    training_features=total_features[0:num_train]
    test_features=total_features[num_train:num_total]
    training_labels=total_labels[0:num_train]
    test_labels=total_labels[num_train:num_total]
    #pack into a class variable to  return and pass around
    ml_data=EmptyClass()
    ml_data.training_features=training_features
    ml_data.test_features=test_features
    ml_data.training_labels=training_labels
    ml_data.test_labels=test_labels
    return ml_data
    
#Compute Receiver/operator curves for evaluating model
def calc_ROC(y_score,test_labels,num_thresh):
#Calculates ROC curve by varying the threshold for the classification model's
#y score at which a Patent is classified as "granted." Each threshold returns
#a True Positive Rate and False Positive Rate which is used to construct ROC 
#curve
    ROC_TPR=[0]*num_thresh
    ROC_FPR=[0]*num_thresh
    predictions = [0]*len(y_score)
    thresh_array=np.linspace(0,1,num=num_thresh)


    for j in range(0,len(thresh_array)):
        for i in range(0,len(y_score)):
            predictions[i]=y_score[i][1]>=thresh_array[j]
        ROC_TPR[j]=(sum(test_labels & predictions) / sum(test_labels))
        ROC_FPR[j]=(sum( (np.logical_not(test_labels)) & predictions) / sum(np.logical_not(test_labels)))
    return [ROC_TPR,ROC_FPR];
    
def calc_metrics(y_score, test_labels,num_thresh):
    metrics = EmptyClass()
    predictions=[0]*len(y_score)
    for i in range(0,len(y_score)):
        predictions[i]=y_score[i][1]>=0.5
    TP =sum(test_labels & predictions) 
    FP = sum(np.logical_not(test_labels) & predictions)
    TN = sum(np.logical_not(predictions) & np.logical_not(test_labels))
    FN = sum(predictions & np.logical_not(test_labels))
    accuracy = (TP + TN)/(TP + FP + FN + TN)
    TPR = TP /(TP + FN)
    precision = TP/(TP + FP)
    TNR = TN / (TN + FP)
    F1 = 2*TPR*precision / (TPR  +precision)
    ROC = calc_ROC(y_score,test_labels,num_thresh)
    AUC=(np.trapz(np.sort(ROC[0]),np.sort(ROC[1])))
    metrics.AUC=AUC
    metrics.ROC_TPR=ROC[0]
    metrics.ROC_FPR=ROC[1]
    metrics.TP=TP
    metrics.FP=FP
    metrics.TN=TN
    metrics.FP=FP
    metrics.accuracy=accuracy
    metrics.TPR=TPR
    metrics.precision=precision
    metrics.TNR=TNR
    metrics.F1=F1
    metrics.predictions=predictions
    return metrics
    
def get_mean_metrics(metrics,metrics_dummy):
    all_F1 = [x.F1 for x in metrics ]
    all_TPR = [x.TPR for x in metrics ]
    all_TNR = [x.TNR for x in metrics ]
    all_accuracy = [x.accuracy for x in metrics ]
    all_precision = [x.precision for x in metrics ]
    all_AUC = [x.AUC for x in metrics ]
    all_F1_dummy = [x.F1 for x in metrics_dummy ]
    all_TPR_dummy = [x.TPR for x in metrics_dummy ]
    all_TNR_dummy = [x.TNR for x in metrics_dummy ]
    all_accuracy_dummy= [x.accuracy for x in metrics_dummy ]
    all_precision_dummy = [x.precision for x in metrics_dummy ]
    all_AUC_dummy = [x.AUC for x in metrics_dummy ]
    data = np.random.random((6,6))
    data[0][0]=np.round(np.mean(all_TPR),2)
    data[0][1]=np.round(np.mean(all_TPR_dummy),2)
    data[0][2]=np.round(np.min(all_TPR),2)
    data[0][3]=np.round(np.min(all_TPR_dummy),2)
    data[0][4]=np.round(np.max(all_TPR),2)
    data[0][5]=np.round(np.max(all_TPR_dummy),2)
    data[1][0]=np.round(np.mean(all_TNR),2)
    data[1][1]=np.round(np.mean(all_TNR_dummy),2)
    data[1][2]=np.round(np.min(all_TNR),2)
    data[1][3]=np.round(np.min(all_TNR_dummy),2)
    data[1][4]=np.round(np.max(all_TNR),2)
    data[1][5]=np.round(np.max(all_TNR_dummy),2)
    data[2][0]=np.round(np.mean(all_accuracy),2)
    data[2][1]=np.round(np.mean(all_accuracy_dummy),2)
    data[2][2]=np.round(np.min(all_accuracy),2)
    data[2][3]=np.round(np.min(all_accuracy_dummy),2)
    data[2][4]=np.round(np.max(all_accuracy),2)
    data[2][5]=np.round(np.max(all_accuracy_dummy),2)
    data[3][0]=np.round(np.mean(all_precision),2)
    data[3][1]=np.round(np.mean(all_precision_dummy),2)
    data[3][2]=np.round(np.min(all_precision),2)
    data[3][3]=np.round(np.min(all_precision_dummy),2)
    data[3][4]=np.round(np.max(all_precision),2)
    data[3][5]=np.round(np.max(all_precision_dummy),2)
    data[4][0]=np.round(np.mean(all_AUC),2)
    data[4][1]=np.round(np.mean(all_AUC_dummy),2)
    data[4][2]=np.round(np.min(all_AUC),2)
    data[4][3]=np.round(np.min(all_AUC_dummy),2)
    data[4][4]=np.round(np.max(all_AUC),2)
    data[4][5]=np.round(np.max(all_AUC_dummy),2)
    data[5][0]=np.round(np.mean(all_F1),2)
    data[5][1]=np.round(np.mean(all_F1_dummy),2)
    data[5][2]=np.round(np.min(all_F1),2)
    data[5][3]=np.round(np.min(all_F1_dummy),2)
    data[5][4]=np.round(np.max(all_F1),2)
    data[5][5]=np.round(np.max(all_F1_dummy),2)
    return data