import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from perceptron import perceptron
import matplotlib.pyplot as plt
import random
import copy



def do_training_test_split(data, split_by_column, num_splits, split_ratio, random_state):
    """
    Objective:
    ----------
                Split pandas dataset into training and test sets, in a stratified sampling way.

    Parameters:
    -----------
                data            : the pandas dataset to be splitted
                split_by_column : a string, the column name from the dataset whose distribution of values to use 
                                    as basis for the stratified splitting.
                num_splits      : integer number of reshuffling and splitting iterations
                split_ratio     : a float, ratio between dimension of test set and training set
                random_state    : an integer, a seed to use for the random splitting

    Returns:
    --------
                two pandas dataframes: the first is a training set, and the second is the test set.
    """
    stratified_training_set     = []
    stratified_test_set         = []
    split_instance = StratifiedShuffleSplit(n_splits=num_splits, test_size=split_ratio, random_state=random_state)
    for train_index, test_index in split_instance.split(data, data[split_by_column]):
        stratified_training_set     = data.loc[train_index]
        stratified_test_set         = data.loc[test_index]
    return stratified_training_set, stratified_test_set

def plot_data_as_binary(data, 
                        predictive_column_names, 
                        target_column_name, 
                        thresh_pass, 
                        thresh_fail,
                        x_label,
                        y_label,
                        positive_plot_label,
                        negative_plot_label,
                        title,
                        show=True):
    X_positive = data[predictive_column_names[0]].where(data[target_column_name]==thresh_pass)
    X_negative = data[predictive_column_names[0]].where(data[target_column_name]==thresh_fail)
    Y_positive = data[predictive_column_names[1]].where(data[target_column_name]==thresh_pass)
    Y_negative = data[predictive_column_names[1]].where(data[target_column_name]==thresh_fail)

    plt.scatter(X_positive, Y_positive, marker="o", label=positive_plot_label)
    plt.scatter(X_negative, Y_negative, marker="x", label=negative_plot_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    if show:
        #plt.draw()
        plt.show()

def w0_vector(num_features, seed_value=63):
    w_seed    = np.random.RandomState(seed=seed_value)
    return w_seed.normal(loc=0.0, scale=0.1, size=(1 + num_features))

def run_full_analysis(model_obj,
                      methods,
                      epochs_list,
                      w0,
                      X_test_data,
                      Y_test_data,
                      n_decimals=4):
    epochs_update_dict = {}
    Final_weights = {}
    N_methods = len(methods)
    for n in range(N_methods):
        epochs_update_dict[n+1] = []
        Final_weights[n+1] = []
    

    #Z = copy.deepcopy(model_obj)
    indices = [i-1 for i in epochs_list]

    for n_method in methods:
        print("\n")
        print('{:s}'.format('\u0333'.join(" Method # = {}:".format(n_method))))
        #w_vector = w0
        Z = copy.deepcopy(model_obj)
        #print('{:s}'.format('\u0333'.join(" \t # epoch number = {}:".format(epochs_list[n]))))
        Z.initial_weights   = w0
        #print("before number of epochs = ", model_obj.n_epochs)
        Z.n_epochs          = epochs_list[-1]
        #print("after number of epochs = ", model_obj.n_epochs)
        Z.fit_and_print_accuracy(X_test             = X_test_data,
                                             Y_test             = Y_test_data,
                                             w_update_method    = n_method,
                                             n_digits           = n_decimals
                )
        #w_vector = Z.get_optimized_weights()
        epochs_update_dict[n_method]= copy.deepcopy(Z.get_update_per_epoch()[indices])
        Final_weights[n_method]=copy.deepcopy(Z.get_optimized_weights())
        #print("INSIDE Final weights = ", Final_weights)
        #print("INSIDE epochs_update_dict = ", epochs_update_dict)
    return epochs_update_dict, Final_weights

def plot_epoch_updates_per_method(epochs_list,
                                  epochs_update_dict,
                                  methods,
                                  learning_rate,
                                  seed_value,
                                  colors_list,
                                  x_label,
                                  y_label,
                                  title=None,
                                  show=True):
    label_str = "method {}, seed = {}"
    p = [plt.plot(epochs_list, epochs_update_dict[n+1], c=colors_list[n], 
                  label=label_str.format(n+1, seed_value)) for n in range(len(epochs_update_dict))]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if title!=None:
        plt.title("Learning rate = {}".format(learning_rate))
    if show:
        #plt.draw()
        plt.show()

def plot_decision_lines(x_min, 
                        x_max,
                        seed,
                        threshold_value, 
                        Final_weights, 
                        epochs_update_dict,
                        colors_list,
                        x_label,
                        y_label,
                        title=None,
                        show=True):

    #x_min = dataset["eruptions"].min()
    #x_max = dataset["eruptions"].max()
    N = len(epochs_update_dict)
    slope = [-Final_weights[n+1][1]/Final_weights[n+1][2] for n in range(N)]
    intercept = [(threshold_value-Final_weights[n+1][0])/Final_weights[n+1][2] for n in range(N)]
    y_min_array = [slope[n]*x_min + intercept[n] for n in range(N)]
    y_max_array = [slope[n]*x_max + intercept[n] for n in range(N)]
    #plt.scatter(X_severe, Y_severe, marker="o")
    #plt.scatter(X_non_severe, Y_non_severe, marker="x")
    P= [plt.plot([x_min, x_max], [y_min_array[n], y_max_array[n]], c=colors_list[n], 
                 label="method {}, seed = {}".format(n+1, seed )) for n in range(N)]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if title!=None:
        plt.title(title)
    if show:
        #plt.draw()
        plt.show()


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

    file_path = os.path.join("my_perceptron","old_faithful.csv")            #user may want to change this
    dataset = pd.read_csv(file_path)
    strat_training_set, strat_test_set = do_training_test_split(data=dataset,
                                                                          split_by_column="severe",
                                                                          num_splits=1,
                                                                          split_ratio=0.2,
                                                                          random_state=1024)
    stratified_training_set             = strat_training_set.to_numpy()


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

    #quit()

    stratified_training_set_predictors  = stratified_training_set[:,1:3]
    stratified_training_set_labels      = stratified_training_set[:,3]
    strat_training_set                  = []

    stratified_test_set                 = strat_test_set.to_numpy()
    stratified_test_set_predictors      = stratified_test_set[:,1:3]
    stratified_test_set_labels          = stratified_test_set[:,3]
    strat_test_set                      = []

    num_features        = np.shape(stratified_training_set_predictors)[1]
    chosen_random_int   = random.randint(1,10**7)
    eta_value           = 10**(-5)
    w0_                  = w0_vector(num_features=num_features, seed_value=chosen_random_int)
    epochs_list_ = [n for n in range(1,201)]
    model = perceptron(data_predictors=stratified_training_set_predictors,
                       data_labels=stratified_training_set_labels,
                       learning_rate=eta_value,
                       threshold_value=0,
                       thresh_pass=1.0,
                       thresh_fail=-1.0,
                       initial_weights=w0_,
                       random_seed=None,
                       n_epochs=1)
    
    methods_ = [1,2,3]
    
    colors_list_ = ['b', 'r', 'k']

    epochs_update_dict_, Final_weights_ = run_full_analysis(model_obj=model,
                                                            methods=methods_,
                                                            epochs_list=epochs_list_,
                                                            w0=w0_,
                                                            X_test_data=stratified_test_set_predictors,
                                                            Y_test_data=stratified_test_set_labels,
                                                            n_decimals=6)   
    
    #print("epochs_update_dict = ",epochs_update_dict_)
    #print("Final weights =", Final_weights_)
    
    #quit()

    plot_epoch_updates_per_method(epochs_list=epochs_list_,
                                  epochs_update_dict=epochs_update_dict_,
                                  methods=methods_,
                                  learning_rate=model.learning_rate,
                                  seed_value=chosen_random_int,
                                  colors_list=colors_list_,
                                  x_label="n-th epoch",
                                  y_label="# of weight updates",
                                  title=None,
                                  show=True
                                  )
    


    plot_decision_lines(x_min=dataset["eruptions"].min(),
                        x_max=dataset["eruptions"].max(),
                        seed=chosen_random_int,
                        threshold_value=model.threshold_value,
                        Final_weights=Final_weights_,
                        epochs_update_dict=epochs_update_dict_,
                        colors_list=colors_list_,
                        x_label="eruptions (min)",
                        y_label="waiting (min)",
                        title=None,
                        show=False)
    plot_data_as_binary(data=dataset,
                            predictive_column_names=["eruptions", "waiting"],
                            target_column_name="severe",
                            thresh_pass=1,
                            thresh_fail=-1,
                            x_label="eruptions (min)",
                            y_label="waiting (min)",
                            positive_plot_label="severe",
                            negative_plot_label="not severe",
                            title="Severity of eruptions at Old Faithful geyser",
                            show=False
                             )
    plt.show()

    quit()
    """for n in range(8,10):
        print("v_vetor begin = {}".format(w0_vector))
        print('{:s}'.format('\u0333'.join(" # passes = {}:".format(2**n))))
        model = perceptron(data_predictors=stratified_training_set_predictors,
                       data_labels=stratified_training_set_labels,
                       learning_rate=0.1,
                       threshold_value=0,
                       thresh_pass=1.0,
                       thresh_fail=-1.0,
                       initial_weights=w0_vector(random_int, num_features),
                       random_seed=63,
                       n_epochs=2**n)
        model.fit_and_print_accuracy(X_test=stratified_test_set_predictors,
                                     Y_test=stratified_test_set_labels,
                                     w_update_method=2,
                                     n_digits=6)
        print(model.get_optimized_weights())
        model.v_initial_weights=w0_vector"""
    
    #accuracy_list = []

    """def run_model(model_obj):
        print("n_pass = {}".format(model_obj.v_num_passes))
        print("Initial vector = {}".format(model_obj.v_initial_weights))
        #model_obj.v_initial_weights = [-0.21389787,  0.11120612,  0.00358016]
        model_obj.fit_and_print_accuracy(X_test=stratified_test_set_predictors,
                                     Y_test=stratified_test_set_labels,
                                     w_update_method=2,
                                     n_digits=6)
        print(model_obj.get_optimized_weights())
        accuracy_list.append(model_obj.get_accuracy())
        del model_obj
    
    models = [ perceptron(data_predictors=stratified_training_set_predictors,
                       data_labels=stratified_training_set_labels,
                       learning_rate=10**(-6),
                       threshold_value=0,
                       thresh_pass=1.0,
                       thresh_fail=-1.0,
                       initial_weights=w0_vector(),
                       random_seed=63,
                       n_epochs=n+1) for n in range(100) ]
    
    for model in models:
        run_model(model)       

    n_epochs = [n+1 for n in range(100)]
    plt.plot(n_epochs, accuracy_list)
    plt.show()"""
    

    #quit()




"""    total_epochs = 500
    eta_value = 10**(-5)
    Final_weights = []
    for n_method in range(1,1+N_methods):
        w_vector    = w0_vector(random_int, num_features)
        n = 0
        #print("w_vector in-between = {}".format(w_vector))
        #print("w0_vector in-between = {}".format(w0_vector))
        while n in range(total_epochs):
            #print("w_vector at beginning = {}".format(w_vector))
            #print('{:s}'.format('\u0333'.join(" # passes = {}:".format(n_base**n))))
            print('{:s}'.format('\u0333'.join(" # epoch number = {}:".format(n+1))))
            model = perceptron(data_predictors=stratified_training_set_predictors,
                               data_labels=stratified_training_set_labels,
                               learning_rate=eta_value,
                               threshold_value=0,
                               thresh_pass=1.0,
                               thresh_fail=-1.0,
                               initial_weights=w_vector,
                               random_seed=None,
                               n_epochs=1)
            model.fit_and_print_accuracy(X_test=stratified_test_set_predictors,
                                     Y_test=stratified_test_set_labels,
                                     w_update_method=n_method,
                                     n_digits=6)
            accuracy_dict[n_method].append(model.get_update_per_epoch())
            w_vector = model.get_optimized_weights()
            #print("w_vector at end = {}".format(w_vector))
            print("\n")
            n+=1
            print("\n")
        #w_vector = w0_vector
        Final_weights.append(list(w_vector))
        n_method+=1"""

    #print("accuracy_dict[1] = {}".format(np.round(np.array(accuracy_dict[1]),4)))
    #print("accuracy_dict[2] = {}".format(np.round(np.array(accuracy_dict[2]),4)))



"""def plot_decision_lines(x_min, 
                        x_max, 
                        threshold_value, 
                        Final_weights, 
                        epochs_update_dict,
                        colors_list,
                        x_label,
                        y_label,
                        title,
                        show=True):

    #x_min = dataset["eruptions"].min()
    #x_max = dataset["eruptions"].max()
    N = len(epochs_update_dict)
    slope = [-Final_weights[n][1]/Final_weights[n][2] for n in range(N)]
    intercept = [(threshold_value-Final_weights[n][0])/Final_weights[n][2] for n in range(N)]
    y_min_array = [slope[n]*x_min + intercept[n] for n in range(N)]
    y_max_array = [slope[n]*x_max + intercept[n] for n in range(N)]
    #plt.scatter(X_severe, Y_severe, marker="o")
    #plt.scatter(X_non_severe, Y_non_severe, marker="x")
    P= [plt.plot([x_min, x_max], [y_min_array[n], y_max_array[n]], c=colors_list[n]) for n in range(N)]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show:
        plt.show()"""

if __name__ == "__main__":
    main()