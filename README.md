# Patent-Classification
A simple demo in predicting patent results

In this code I load the JSON patent data, clean it, divide it into training and testing sets, and apply a Random Forest Classifier to classify whether a patent will be granted based on all the metadata attached it. Finally, and most importantly, the classification is validated using the F1 score compared to the F1 score given by a variety of dummy classifiers. 

main_script.py - The script which trains and tests the model. The script can also be used to test the model on another JSON file

main_functions.py - The functions for training, testing, and evaluating the model

helper_functions.py - the helper functions for main_functions.py, and also for loading, cleaning, and dividing the data into test and train sets

evaluate_model_for_permuted_data.py - permutes the data before taking test and train sets, taking a total of 100 random shufflings. the metrics are computed for each shuffling for analysis.

generate_plots.py - the code to generate the final plots and table for the report


