import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from perceptron import perceptron
import matplotlib.pyplot as plt
import random



def do_training_test_split(data, split_by_column, num_splits, split_ratio, random_state):
    """
    Objective:
                Split pandas dataset into training and test sets, in a stratified sampling way.

    Parameters:
                data            : the pandas dataset to be splitted
                split_by_column : a string, the column name from the dataset whose distribution of values to use 
                                    as basis for the stratified splitting.
                num_splits      : integer number of reshuffling and splitting iterations
                split_ratio     : a float, ratio between dimension of test set and training set
                random_state    : an integer, a seed to use for the random splitting

    Returns:
                two pandas dataframes: the first is a training set, and the second is the test set.
    """
    stratified_training_set     = []
    stratified_test_set         = []
    split_instance = StratifiedShuffleSplit(n_splits=num_splits, test_size=split_ratio, random_state=random_state)
    for train_index, test_index in split_instance.split(data, data[split_by_column]):
        stratified_training_set     = data.loc[train_index]
        stratified_test_set         = data.loc[test_index]
    return stratified_training_set, stratified_test_set


def main():
    #print(os.getcwd())
    file_path = os.path.join("my_perceptron","old_faithful.csv")
    dataset = pd.read_csv(file_path)
    strat_training_set, strat_test_set = do_training_test_split(data=dataset,
                                                                          split_by_column="severe",
                                                                          num_splits=1,
                                                                          split_ratio=0.2,
                                                                          random_state=1024)
    stratified_training_set             = strat_training_set.to_numpy()
    stratified_training_set_predictors  = stratified_training_set[:,1:3]
    stratified_training_set_labels      = stratified_training_set[:,3]
    strat_training_set                  = []

    stratified_test_set                 = strat_test_set.to_numpy()
    stratified_test_set_predictors      = stratified_test_set[:,1:3]
    stratified_test_set_labels          = stratified_test_set[:,3]
    strat_test_set                      = []


    m = random.randint(1,10**7)
    #m = 2
    def v_vector(s=63):
        w_seed    = np.random.RandomState(seed=s)
        return w_seed.normal(loc=0.0, scale=0.1, size=1 + np.shape(stratified_training_set_predictors)[1])
    
    print("v_vector = {}".format(v_vector))
    """for n in range(8,10):
        print("v_vetor begin = {}".format(v_vector))
        print('{:s}'.format('\u0333'.join(" # passes = {}:".format(2**n))))
        model = perceptron(data_predictors=stratified_training_set_predictors,
                       data_labels=stratified_training_set_labels,
                       learning_rate=0.1,
                       threshold_value=0,
                       thresh_pass=1.0,
                       thresh_fail=-1.0,
                       initial_weights=v_vector,
                       random_seed=63,
                       n_epochs=2**n)
        model.fit_and_print_accuracy(X_test=stratified_test_set_predictors,
                                     Y_test=stratified_test_set_labels,
                                     w_update_method=2,
                                     n_digits=6)
        print(model.get_optimized_weights())
        model.v_initial_weights=v_vector"""
    
    accuracy_list = []

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
                       initial_weights=v_vector(),
                       random_seed=63,
                       n_epochs=n+1) for n in range(100) ]
    
    for model in models:
        run_model(model)       

    n_epochs = [n+1 for n in range(100)]
    plt.plot(n_epochs, accuracy_list)
    plt.show()"""
    

    #quit()

    accuracy_dict = {
        1   : [],
        2   : [],
        3   : []
    }

    max_power = 500
    n_base = 2
    n_method = 1
    eta_value = 10**(-6)
    while n_method in [1,2,3]:
        w_vector    = v_vector(m)
        print("m seed = ", m)
        n = 0
        #print("w_vector in-between = {}".format(w_vector))
        #print("V_vector in-between = {}".format(v_vector))
        while n in range(max_power):
            #print("w_vector at beginning = {}".format(w_vector))
            #print('{:s}'.format('\u0333'.join(" # passes = {}:".format(n_base**n))))
            print('{:s}'.format('\u0333'.join(" # passes = {}:".format(n+1))))
            #c = n_base**n - n_base**(n-1) if n>0 else 1
            c = n+1
            print("c = {}".format(c))
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
            #cur_accuracy = model.get_accuracy()
            #updated_accuracy = (n_base**(n-1)*accuracy_dict[n_method][n-1] + c*cur_accuracy)/(n_base**n) if n>0 else cur_accuracy
            #accuracy_dict[n_method].append(model.get_accuracy())
            accuracy_dict[n_method].append(model.get_update_per_epoch())
            w_vector = model.get_optimized_weights()
            #print("w_vector at end = {}".format(w_vector))
            print("\n")
            n+=1
            print("\n")
        #w_vector = v_vector
        n_method+=1

    #print("accuracy_dict[1] = {}".format(np.round(np.array(accuracy_dict[1]),4)))
    #print("accuracy_dict[2] = {}".format(np.round(np.array(accuracy_dict[2]),4)))

    #n_epochs = [n_base**n for n in range(max_power)]
    n_epochs = [n+1 for n in range(max_power)]
    plt.plot(n_epochs, accuracy_dict[1], c='b', label='method 1, seed = {}'.format(m))
    plt.plot(n_epochs, accuracy_dict[2], c='r', label='method 2, seed = {}'.format(m))
    plt.plot(n_epochs, accuracy_dict[3], c='k', label='method 3, seed = {}'.format(m))
    plt.xlabel("n-th epoch")
    plt.ylabel("No. weight updates")
    plt.legend()
    plt.title("eta = {}".format(eta_value))
    plt.show()


if __name__ == "__main__":
    main()