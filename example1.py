import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from perceptron import perceptron



def do_training_test_split(data, split_by_column, num_splits, split_ratio, random_state):
    stratified_training_set     = []
    stratified_test_set         = []
    split_instance = StratifiedShuffleSplit(n_splits=num_splits, test_size=split_ratio, random_state=random_state)
    for train_index, test_index in split_instance.split(data, data[split_by_column]):
        #print("len(train_index) = {}".format(len(train_index)))
        #print("train_index = {}".format(train_index))
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
    #print(dataset.head())
    stratified_training_set             = strat_training_set.to_numpy()
    stratified_training_set_predictors  = stratified_training_set[:,1:3]
    stratified_training_set_labels      = stratified_training_set[:,3]
    strat_training_set                  = []
    #print("stratified_training_set_predictors[-1]:")
    #print(stratified_training_set_predictors[-1])
    #quit()

    stratified_test_set                 = strat_test_set.to_numpy()
    stratified_test_set_predictors      = stratified_test_set[:,1:3]
    stratified_test_set_labels          = stratified_test_set[:,3]
    strat_test_set                      = []

    weights = []
    for n in range(3):
        print("n = {}".format(n))
        model = perceptron(data_predictors=stratified_training_set_predictors,
                       data_labels=stratified_training_set_labels,
                       eta=0.1,
                       threshold_value=0,
                       thresh_pass=1.0,
                       thresh_fail=-1.0,
                       initial_weights=None,
                       random_seed=1,
                       n_passes=5**n)
        model.fit_and_print_accuracy(X_test=stratified_test_set_predictors,
                                     Y_test=stratified_test_set_labels,
                                     n_digits=6)
        print("\n")
        print("\n")


if __name__ == "__main__":
    main()