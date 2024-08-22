import numpy as np
import pandas as pd
import os
from perceptron import perceptron
import matplotlib.pyplot as plt
from job_utility import run_full_analysis, w0_vector, do_training_test_split, plot_SSE_per_epoch
from job_utility import plot_data_as_binary, plot_decision_lines, plot_epoch_updates_per_method



def main():
    """
    Objective:
    ----------
                In this example, we use the perceptron algorithm to classify whether an eruption
                at the Old Faithful geyser in Yellowstone National Park, Wyoming, USA, is "severe" (coded as +1)
                or "not severe" (coded as -1).

    Description of the dataset
    --------------------------
                The dataset has the following columns:
                    # Column        Column name         dytpe           Description \n
                        0           sample              numeric         Measurement number \n
                        1           eruptions           numeric         Eruption time in mins \n
                        2           waiting             numeric         Waiting time to next eruption in mins \n
                        3           severe              categorical     whether the eruption was "severe" (+1) or not (-1). \n
                Dataset was downloaded from https://cs.colby.edu/courses/F22/cs343/projects/p1adaline/p1adaline.html.
    """

    file_path = os.path.join("old_faithful.csv")            #user may want to change this
    dataset = pd.read_csv(file_path)

    # do a binary plot of the dataset. The binary category column is named "severe", 
    # and values are either +1 or -1.
    """P1 = plot_data_as_binary(data=dataset,
                            predictive_column_names=["eruptions", "waiting"],
                            target_column_name="severe",
                            thresh_pass=1,
                            thresh_fail=-1,
                            x_label="eruptions",
                            y_label="waiting",
                            positive_plot_label="severe",
                            negative_plot_label="not severe",
                            title="Severity of eruptions at Old Faithful geyser"
                             )"""
    
    # do a stratified splitting of the dataset into training ans test sets, using the column "severe" as base
    # for the stratified splitting.
    strat_training_set, strat_test_set  = do_training_test_split(data           = dataset,
                                                                split_by_column = "severe",
                                                                num_splits      = 1,
                                                                split_ratio     = 0.2,
                                                                random_state    = 1024)

    # transform training set into numpy array and split into predictors and target data
    stratified_training_set             = strat_training_set.to_numpy()
    stratified_training_set_predictors  = stratified_training_set[:,1:3]
    stratified_training_set_labels      = stratified_training_set[:,3]
    strat_training_set                  = []

    # transform test set into numpy array and split into predictors and target data
    stratified_test_set                 = strat_test_set.to_numpy()
    stratified_test_set_predictors      = stratified_test_set[:,1:3]
    stratified_test_set_labels          = stratified_test_set[:,3]
    strat_test_set                      = []

    # setting additional parameters before running the perceptron
    # eta_value is the learning rate
    # w0_ is the initial weight vector
    # epochs_list_ is the list of epoch numbers we are interested in
    num_features            = np.shape(stratified_training_set_predictors)[1]
    random_int_array        = np.random.randint(low=1,high=10**7,size=10)
    chosen_random_int       = np.random.choice(random_int_array, size=1)[0]
    random_int_array        = None
    eta_value               = 10**(-5)
    w0_                     = w0_vector(num_features=num_features, seed_value=chosen_random_int) 
    epochs_list_            = [n for n in range(1,501)]

    # creating a perceptron object with our data and chosen parameters
    model = perceptron(data_predictors      = stratified_training_set_predictors,
                       data_labels          = stratified_training_set_labels,
                       learning_rate        = eta_value,
                       threshold_value      = 0,
                       thresh_pass          = 1.0,
                       thresh_fail          = -1.0,
                       initial_weights      = w0_,
                       random_seed          = None,
                       n_epochs             = 1)
    
    methods_        = [1,2,3]               # integer codes for weight updating methods  
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
                                                                         n_decimals  = 6)   


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
    


    plot_decision_lines(x_min                   = dataset["eruptions"].min(),
                        x_max                   = dataset["eruptions"].max(),
                        seed                    = chosen_random_int,
                        threshold_value         = model.threshold_value,
                        Final_weights           = Final_weights_,
                        epochs_update_dict      = epochs_update_dict_,
                        colors_list             = colors_list_,
                        x_label                 = "eruptions (min)",
                        y_label                 = "waiting (min)",
                        title                   = None,
                        show                    = False
                        )
    
    plot_data_as_binary(data                    = dataset,
                        predictive_column_names = ["eruptions", "waiting"],
                        target_column_name      = "severe",
                        thresh_pass             = 1,
                        thresh_fail             = -1,
                        x_label                 = "eruptions (min)",
                        y_label                 = "waiting (min)",
                        positive_plot_label     = "severe",
                        negative_plot_label     = "not severe",
                        title                   = "Severity of eruptions at Old Faithful geyser",
                        show                    = False
                        )
    plt.show()

    plot_SSE_per_epoch(SSE_dict                 = SSE_vector_,
                       seed                     = chosen_random_int,
                       epochs_list              = epochs_list_,
                       colors_list              = colors_list_,
                       x_label                  = "n-th epoch",
                       y_label                  = "SSE",
                       title                    = "Sum of squared errors (SSE) vs epoch",
                       show                     = True
                       )
    

if __name__ == "__main__":
    main()