#This is the code used to generate plots and tables in the .pdf document 


#Convert array of structs into separate arrays for each metric
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

#Plot histograms of F1 scores
pyplot.hist(all_F1_dummy,alpha=0.5,label="Dummy classifier")
pyplot.hist(all_F1,alpha=0.5,label="Random Forest")
pyplot.title('Histogram of F1 scores')
pyplot.xlabel('F1 score')
pyplot.ylabel('Frequency')
pyplot.legend()
pyplot.show()

#Plot histograms of AUC-ROC scores
pyplot.hist(all_AUC_dummy,alpha=0.5,label="Dummy classifier")
pyplot.hist(all_AUC,alpha=0.5,label="Random Forest")
pyplot.title('Histogram of AUC scores')
pyplot.xlabel('AUC score')
pyplot.ylabel('Frequency')
pyplot.legend()
pyplot.show()

#Create a table of all metrics
rows = ('TPR','TNR','Accuracy','Precision','AUC','F1')
columns = ['RF Mean','Dummy Mean','RF Min','Dummy Min','RF Max','Dummy max']
data=get_mean_metrics(metrics,metrics_dummy)

the_table=pyplot.table(cellText=data,rowLabels=rows,colLabels=columns,loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1.5,1.5)

# Removing ticks and spines enables you to get the figure only with table
pyplot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
pyplot.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
for pos in ['right','top','bottom','left']:
    pyplot.gca().spines[pos].set_visible(False)
    
pyplot.show()