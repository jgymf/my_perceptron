import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import perceptron

file_path = os.path.join("my_perceptron","Iris.csv")
dataset = pd.read_csv(file_path)
#print(dataset.head())
#print(dataset.describe())
#print(dataset.info())
#print(dataset["Species"].value_counts())

#The first 50 entries of the dataset are classified as 'Iris-setosa'
#The subsequent 50 entries are classified as 'Iris-versicolor'
#The last 50 entries are 'Iris-virginica'
#We want to do a perceptron training on the first 100 entries, so we have a binary classification.
col_with_labels = 5
begin_row_num   = 0
end_row_num     = 100
Y_labels = dataset.iloc[begin_row_num:end_row_num, col_with_labels].values
#print(Y_labels)
#Feature scaling: We convert the string 'Iris-setosa' into integer -1, and 'Iris-versicolor' into 1
Y_labels = np.where(Y_labels=='Iris-setosa', -1, 1)
#print(Y_labels)

#Assume the classicafication may be accurately done by just looking at the sepal length (column # = 1)
#  and petal length (column # = 3)
predictors_column_num = [1,3]
X_predictors = dataset.iloc[begin_row_num:end_row_num, predictors_column_num].to_numpy()
n_epochs = 20 

model = perceptron(data_predictors=X_predictors,
                   data_labels=Y_labels,
                   learning_rate=10**(-4),
                   threshold_value=0,
                   thresh_pass=1,
                   thresh_fail=-1,
                   random_seed=1,
                   n_epochs=n_epochs)
model.fit(w_update_method=1)
success_rate = model.get_update_per_epoch()
epochs = [n for n in range(1,n_epochs+1)]

#print(model.learning_rate)

#quit()
plt.plot(epochs, success_rate, marker='o')
plt.xlabel("n-th epochs")
plt.ylabel("# of weight updates")
plt.show()

