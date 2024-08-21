import numpy as np
import pandas as pd
import os
from adaline import adaline
import matplotlib.pyplot as plt
from job_utility import run_full_analysis, w0_vector, do_training_test_split, standardize, plot_SSE_per_epoch
from job_utility import plot_data_as_binary, plot_decision_lines, plot_epoch_updates_per_method


def main():
    """
    Objective:
    ----------
                Employ the perceptron algorithm to tell whether an iris is an "Iris setosa" or "Iris-versicolo".

    Description of the dataset
    --------------------------
                The dataset has the following columns:
                    # Column        Column name         dytpe   \n
                        0           Id                  numeric \n
                        1           SepalLengthCm       numeric \n
                        2           SepalWidthCm        numeric \n
                        3           PetalLengthCm       numeric \n
                        4           PetalWidthCm        numeric \n
                        5           Species             string  \n
                Dataset was downloaded from https://www.kaggle.com/datasets/uciml/iris.
    """    

    file_path = os.path.join("Iris.csv")
    Dataset = pd.read_csv(file_path)
    print(Dataset.head())
    print("\n")
    print(Dataset.describe())
    print("\n")
    print(Dataset.info())
    print("\n")
    #print(dataset["Species"].value_counts())


    """
    The first 50 entries of the dataset are classified as 'Iris-setosa'
    The subsequent 50 entries are classified as 'Iris-versicolor'
    The last 50 entries are 'Iris-virginica'
    We want to do a perceptron training on the first 100 entries, so we have a binary classification.
    We create a copy of that slice below to avoid SettingWIthCopy warnings.
    We also convert the string 'Iris-setosa' into integer -1, and 'Iris-versicolor' into 1.
    """
    dataset = Dataset.head(100).copy()
    dataset["Species"].replace({'Iris-setosa':-1, 'Iris-versicolor':1}, inplace=True)
    #print(dataset["Species"])

    # do a stratified splitting of the dataset into training ans test sets, using the column "severe" as base
    # for the stratified splitting.
    strat_training_set, strat_test_set  = do_training_test_split(data           = dataset,
                                                                split_by_column = "Species",
                                                                num_splits      = 1,
                                                                split_ratio     = 0.2,
                                                                random_state    = 443)
    
    #define predictor columns ('SepalLengthCm' (col 1) and 'PetalLengthCm' (col 3)), and target column ('Species' (col 5))
    predictors_column_num               = [1,3]
    target_column_num                   = 5

    # transform training set into numpy array and split into predictors and target data
    stratified_training_set             = strat_training_set.to_numpy()

    stratified_training_set_predictors  = standardize(stratified_training_set[:,predictors_column_num])
    stratified_training_set_labels      = stratified_training_set[:,target_column_num]
    strat_training_set                  = []

    # transform test set into numpy array and split into predictors and target data
    stratified_test_set                 = strat_test_set.to_numpy()
    stratified_test_set_predictors      = standardize(stratified_test_set[:,predictors_column_num])
    stratified_test_set_labels          = stratified_test_set[:,target_column_num]
    strat_test_set                      = []

    # setting additional parameters before running the perceptron
    # eta_value is the learning rate
    # w0_ is the initial weight vector
    # epochs_list_ is the list of epoch numbers we are interested in
    num_features            = np.shape(stratified_training_set_predictors)[1]
    random_int_array        = np.random.randint(low=1,high=10**7,size=10)
    chosen_random_int       = np.random.choice(random_int_array, size=1)[0]
    #chosen_random_int       = 1
    random_int_array        = None
    eta_value               = 10**(-3)
    w0_                     = w0_vector(num_features=num_features, seed_value=chosen_random_int) 
    epochs_list_            = [n for n in range(1,101)]

    # creating a perceptron object with our data and chosen parameters
    model = adaline(data_predictors      = stratified_training_set_predictors,
                    data_labels          = stratified_training_set_labels,
                    learning_rate        = eta_value,
                    threshold_value      = 0,
                    thresh_pass          = 1.0,
                    thresh_fail          = -1.0,
                    initial_weights      = w0_,
                    random_seed          = None,
                    n_epochs             = 1
                    )

    methods_        = [1,2]               # integer codes for weight updating methods  
    colors_list_    = ['b', 'r', 'k']       # colors for the various weight updating methods

    # get epochs_update_dict_ (a dicitionary recording how many times the weight vector was updated for
    # each interested epoch number, and for each method), and 
    # Final_weights_ (a list of 1D numpy arrays, where each array is the final weight vector learned 
    # from the perceptron for a particlar method) after running the perceptron
    epochs_update_dict_, Final_weights_, SSE_vector_ = run_full_analysis(model_obj   = model,
                                                                         methods     = methods_,
                                                                         epochs_list = epochs_list_,
                                                                         w0          = w0_,
                                                                         X_test_data = stratified_test_set_predictors,
                                                                         Y_test_data = stratified_test_set_labels,
                                                                         n_decimals  = 6
                                                                         )  
    
    plot_epoch_updates_per_method(epochs_list       = epochs_list_,
                                  epochs_update_dict= epochs_update_dict_,
                                  learning_rate     = model.learning_rate,
                                  seed_value        = chosen_random_int,
                                  colors_list       = colors_list_,
                                  x_label           = "n-th epoch",
                                  y_label           = "# of weight updates",
                                  title             = None,
                                  show              = True
                                  )

    plot_decision_lines(x_min                   = standardize(dataset["SepalLengthCm"].to_numpy()).min(),
                        x_max                   = standardize(dataset["SepalLengthCm"].to_numpy()).max(),
                        seed                    = chosen_random_int,
                        threshold_value         = model.threshold_value,
                        Final_weights           = Final_weights_,
                        epochs_update_dict      = epochs_update_dict_,
                        colors_list             = colors_list_,
                        x_label                 = "SepalLength (cm)",
                        y_label                 = "PetalLength (cm)",
                        title                   = None,
                        show                    = False)
    
    plot_data_as_binary(data                    = dataset,
                        predictive_column_names = ["SepalLengthCm", "PetalLengthCm"],
                        target_column_name      = "Species",
                        thresh_pass             = 1,
                        thresh_fail             = -1,
                        x_label                 = "SepalLength (cm)",
                        y_label                 = "PetalLength (cm)",
                        positive_plot_label     = "Iris-versicolor",
                        negative_plot_label     = "Iris-setosa",
                        title                   = "Classification of Iris into Iris-setosa and Iris-versicolor",
                        do_standardize          = True,
                        show                    = False
                        )
    plt.show()

    plot_SSE_per_epoch(SSE_dict                 = SSE_vector_,
                       seed                     = chosen_random_int,
                       epochs_list              = epochs_list_,
                       colors_list              = colors_list_,
                       x_label                  = "n-th epoch",
                       y_label                  = "SSE",
                       title                    = "Sum of Squared Errors (SSE) vs. epoch",
                       show                     = True
                       )

if __name__=="__main__":
    main()