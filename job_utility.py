import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import copy
import pandas as pd

def standardize(np_dataset):
    np_dataset      = np.squeeze(np_dataset)
    dataset_        = np.copy(np_dataset)
    if dataset_.ndim == 1:
        dataset_ = (np_dataset - np_dataset.mean())/np_dataset.std()
    else:
        num_columns = np.shape(dataset_)[1]
        for n_col in range(num_columns):
            dataset_[:,n_col] = np.divide(np.subtract(np_dataset[:,n_col], np_dataset[:,n_col].mean()),np_dataset[:,n_col].std())
    return dataset_

def standardize_pandas(pd_data, list_column_names):
    c_data              = pd_data.copy(deep=True)
    for col in list_column_names:
        mean            = pd_data[col].mean()
        std             = pd_data[col].std()
        c_data[col]     = pd_data[col].transform(lambda x: (x-mean)/std)
    return c_data

def run_full_analysis(model_obj,
                      methods,
                      epochs_list,
                      w0,
                      X_test_data,
                      Y_test_data,
                      n_decimals=4):
    epochs_update_dict          = {}
    Final_weights               = {}
    SSE_vector                  = {}
    N_methods                   = len(methods)

    for n in range(N_methods):
        epochs_update_dict[n+1] = []
        Final_weights[n+1]      = []
        SSE_vector[n+1]         = []
    
    # create a list of indices corresponding to the epoch numbers.
    # subtracting 1 because indexing in Python starts at 0. 
    indices = [i-1 for i in epochs_list]

    for n_method in methods:
        print("\n")
        print('{:s}'.format('\u0333'.join(" Method # = {}:".format(n_method))))
        # make deep copy of object
        Z = copy.deepcopy(model_obj)
        Z.initial_weights   = w0
        #print("Initial weights = ", Z.initial_weights)
            
        Z.n_epochs          = epochs_list[-1]
        # train perceptron on data, test the final weights on the test set and print accuracy to console
        Z.fit_and_print_accuracy(X_test         = X_test_data,
                                Y_test          = Y_test_data,
                                w_update_method = n_method,
                                n_digits        = n_decimals
                                )
        # get number of times the weight vector was updated for the interested epoch numbers 
        # and store in epochs_update_dict for the corresponding method number.
        #print("get_update_per_epoch = ", Z.get_update_per_epoch())
        epochs_update_dict[n_method]= copy.deepcopy(Z.get_update_per_epoch()[indices])
        Final_weights[n_method]     = copy.deepcopy(Z.get_optimized_weights())
        SSE_vector[n_method]        = copy.deepcopy(Z.get_SSE_per_epoch())
    return epochs_update_dict, Final_weights, SSE_vector


def w0_vector(num_features, seed_value=63):
    """
    Objective:
    ---------
                Generate a numpy 1D array of random float numbers to be used to initialize the perceptron weight vector.
                Random numbers are drawn from a normal distribution centered on zero, with starndard deviation of 0.1.
                The size of the array depends on the number of features being taken into consideration.

    Parameters:
    -----------
                num_features    : an integer, denotes the number of predictive features\n
                seed_value      : an integer, seed to be used to randomly generate the entries in the initial weight vector\n

    Returns:
    --------
                a 1D numpy array of dimension (1+num_features).
    """
    w_seed    = np.random.RandomState(seed=seed_value)
    return w_seed.normal(loc=0.0, scale=0.1, size=(1 + num_features))


def do_training_test_split(data, split_by_column, num_splits, split_ratio, random_state):
    """
    Objective:
    ----------
                Split pandas dataset into training and test sets, in a stratified sampling way.

    Parameters:
    -----------
                data            : the pandas dataset to be splitted\n
                split_by_column : a string, the column name from the dataset whose distribution of values to use\n
                                    as basis for the stratified splitting\n
                num_splits      : integer number of reshuffling and splitting iterations\n
                split_ratio     : a float, ratio between dimension of test set and training set\n
                random_state    : an integer, a seed to use for the random splitting\n

    Returns:
    --------
                two pandas dataframes: the first is a training set, and the second is the test set.
    """
    stratified_training_set         = []
    stratified_test_set             = []
    split_instance                  = StratifiedShuffleSplit(n_splits=num_splits, 
                                                             test_size=split_ratio, 
                                                             random_state=random_state)
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
                        do_standardize=False,
                        show=True):
    """
    Objective:
    ---------
                Generate a binary plot of a given dataset. 
                (Number of features to consider must be three, including the binary categorical feature.)

    Parameters:
    ----------
                data                    : a pandas dataframe\n
                predictive_column_names : list, names of columns to use as predictive features\n
                thresh_pass             : any, the upper categotical value\n
                thresh_fail             : any, the lower categorical value\n
                x_label                 : a string, label for the x-axis\n
                y_label                 : a string, label for the y-axis\n
                positive_plot_label     : a string, title for the categorical positive data points in the legend\n
                negative_plot_label     : a string, title for the categorical negative data points in the legend\n
                title                   : a string, title of the plot. Default is None\n
                show                    : boolean, True (default) if one wants to plot to pop up immediately. Else False. 

    Returns:
    --------
                None
    """
    if do_standardize:
        data        = standardize_pandas(pd_data            = data, 
                                         list_column_names  = predictive_column_names)
    data_positive   = data[predictive_column_names].where(data[target_column_name]==thresh_pass).dropna().to_numpy()
    data_negative   = data[predictive_column_names].where(data[target_column_name]==thresh_fail).dropna().to_numpy()
    plt.scatter(data_positive[:,0], data_positive[:,1], marker="o", label=positive_plot_label)
    plt.scatter(data_negative[:,0], data_negative[:,1], marker="x", label=negative_plot_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if title!=None:
        plt.title(title)
    if show:
        plt.show()


    
def plot_epoch_updates_per_method(epochs_list,
                                  epochs_update_dict,
                                  learning_rate,
                                  seed_value,
                                  colors_list,
                                  x_label,
                                  y_label,
                                  title=None,
                                  show=True):
    """
    Objective:
    ----------
                Generates a plot of how many times the weight vector was updated for a given list of epoch numbers.

    Parameters:
    -----------
                epochs_list         : a list of integers representing specific epoch numbers and 
                                        their corresponding number of weight vector updates we want to plot to plot\n
                epochs_update_dict  : a dictionary, keys are integers (i.e. 1,2,3) representing a method for
                                        updating the weight vector. The values of this dictionary are 1D numpy arrays of postive integers.
                                        Each value of the array represents the number of times the weight vector was updated
                                        when training the perceptron on the dataset at a particular epoch number.
                                        Note that len(epochs_list) = len(epochs_update_dict[n]), with n = {1,2,3}. 
                                        For example, say,
                                        epochs_list             =          [5,  10, 15, 20, 25, 30, 35]
                                        epochs_update_dict[2]   = np.array([90, 76, 37, 10,  8,  0,  0]).
                                        This means that with "method 2" for weight vector updating, the weight vector was updated 
                                        90 times at epoch #5, 76 times at epoch #10, 37 times at epoch #15, and so on.\n
                learning_rate       : a float, usually between 0 and 1\n
                seed_value          : an integer, seed used to randomly generate the initial weight vector\n
                colors_list         : a list of strings representing colors. Length of string should coincide with len(epochs_update_dict)\n
                x_label             : a string, title for x-axis\n
                y_label             : a string, title for y-axis\n
                title               : a string, title for whole plot. Default is None\n
                show                : boolean, if True plot will pop up. Default is True\n

    Returns:
    --------
                None

    """
    label_str   = "method {}, seed = {}"
    p           = [plt.plot(epochs_list, epochs_update_dict[n+1], c=colors_list[n], 
                  label=label_str.format(n+1, seed_value)) for n in range(len(epochs_update_dict))]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if title!=None:
        plt.title("Learning rate = {}".format(learning_rate))
    if show:
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
    """
    Objective:
    ----------
                Plot decision boundary line for a perceptron training based on TWO predictive features.
                Three lines are generated, one for each method of updating the weight vector.

    Parameters:
    -----------
                x_min               : a number, minimum value of predictive feature on the x-axis in the original dataset\n
                x_max               : a number, maximum value of predictive feature on the x-axis in the original dataset \n
                seed                : an integer, seed used to randomly generate the initial weight vector\n
                threshold_value     : a float, represents boundary value of the step function in evaluating the net input\n
                Final_weights       : a list of 1D numpy arrays; dimension of list coincides with the number of weight updating methods,
                                        and each numpy array represents the final weight vector learned from running the perceptron.
                                        Note that the len of each 1D numpy array is the same and equals (#predictive features +1)\n
                epochs_update_dict  :a dictionary, keys are integers (i.e. 1,2,3) representing a method for
                                        updating the weight vector. The values of this dictionary are 1D numpy arrays of postive integers.
                                        Each value of the array represents the number of times the weight vector was updated
                                        when training the perceptron on the dataset at a particular epoch number.
                                        Note that len(epochs_list) = len(epochs_update_dict[n]), with n = {1,2,3}. 
                                        For example, say,
                                        epochs_list             =          [5,  10, 15, 20, 25, 30, 35]
                                        epochs_update_dict[2]   = np.array([90, 76, 37, 10,  8,  0,  0]).
                                        This means that with "method 2" for weight vector updating, the weight vector was updated 
                                        90 times at epoch #5, 76 times at epoch #10, 37 times at epoch #15, and so on.\n
                colors_list         : a list of strings representing colors. Length of string should coincide with len(epochs_update_dict)\n
                x_label             : a string, title for x-axis\n
                y_label             : a string, title for y-axis\n
                title               : a string, title for whole plot. Default is None\n
                show                : boolean, if True plot will pop up. Default is True\n

    Returns:
    --------
                None
    """
    N           = len(epochs_update_dict)
    slope       = [-Final_weights[n+1][1]/Final_weights[n+1][2] for n in range(N)]
    intercept   = [(threshold_value-Final_weights[n+1][0])/Final_weights[n+1][2] for n in range(N)]
    y_min_array = [slope[n]*x_min + intercept[n] for n in range(N)]
    y_max_array = [slope[n]*x_max + intercept[n] for n in range(N)]
    P= [plt.plot([x_min, x_max], [y_min_array[n], y_max_array[n]], c=colors_list[n], 
                 label="method {}, seed = {}".format(n+1, seed )) for n in range(N)]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if title!=None:
        plt.title(title)
    if show:
        plt.show()

def plot_SSE_per_epoch(SSE_dict,
                        seed,
                        epochs_list,
                        colors_list,
                        x_label,
                        y_label,
                        title=None,
                        show=True):
    N = len(SSE_dict)
    dummy = [plt.plot(epochs_list, SSE_dict[n+1], c=colors_list[n], marker = 'o',
                          label="method {}, seed = {}".format(n+1, seed)) for n in range(N)]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if title!=None:
        plt.title(title)
    if show:
        plt.show()