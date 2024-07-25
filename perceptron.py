import numpy as np
import random
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit


class perceptron:
    def __init__(self, 
                 data_predictors,
                 data_labels,
                 eta, 
                 convergence_rate,
                 threshold_value,
                 thresh_pass=1.0,
                 thresh_fail=-1.0,
                 initial_weights=None,
                 random_seed=None,
                 n_passes=20):
        self.v_data_predictors  = data_predictors
        self.v_current_weights  = initial_weights
        #self.v_previous_weights = None
        #self.v_X                = None
        self.v_y_label          = data_labels
        #self.v_y_predicted      = None
        self.random_seed        = random_seed
        self.v_eta              = eta
        self.v_convergence_rate = convergence_rate
        self.v_threshold_value  = threshold_value
        self.v_num_rows         = None
        self.v_num_columns      = None
        self.v_thresh_pass      = thresh_pass
        self.v_thresh_fail      = thresh_fail
        self.v_random_seed      = random_seed
        self.v_net_input        = None
        self.v_max_iterations   = None
        self.max_abs_error      = None
        self.run_successfully   = False
        self.v_num_passes       = n_passes

    def get_data_dimension(self,X):
        self.v_num_rows         = np.shape(X)[0]
        self.v_num_columns      = np.shape(X)[1]
        #print("row dimension is {}".format(self.v_num_rows))
        #print("column dimension is {}".format(self.v_num_columns))

    def initialize_weights(self):
        if self.v_current_weights is None:
            if self.v_random_seed == None:
                self.v_random_seed = random.randint(1,10**6)
            print("The random seed used to initialize the weight vector is {}\n".format(self.v_random_seed))
            random.seed(self.v_random_seed)
            #self.v_current_weights = np.random.random(self.v_num_columns)
            random_draw  = np.random.RandomState(seed=self.v_random_seed)
            self.v_current_weights  = random_draw.normal(loc=0.0, scale=0.1, size=1+self.v_num_columns)
            #print("This is the initial weight vector:")
            #print(self.v_current_weights)
        if len(self.v_current_weights) != self.v_num_columns + 1:
                print("ERROR: Dimension of initial weight vector does not match dimension of dataset (i.e., number of columns)\n")
                print("\n ... Exiting now.\n")
                exit
        #self.v_previous_weights = np.zeros(shape=self.v_num_columns+1, dtype=float)

    def calculate_net_input(self, w, x):
        #c = np.dot(np.transpose(self.v_current_weights),self.v_data_predictors[iter_num])
        c = np.sum(np.multiply(w[1:],x))
        c += w[0]
        #c = np.sum(np.multiply(self.v_current_weights[1:], self.v_data_predictors[iter_num]))
        #c += self.v_current_weights[0]
        return c
    
    def evaluate_threshold_function(self, z):
        return (self.v_thresh_pass if z>=self.v_threshold_value else self.v_thresh_fail)
    
    def update_weights(self,y_label, y_predicted, x):
        scaled_predict_error        = self.v_eta*(y_label-y_predicted)
        delta_w                     = np.multiply(x, scaled_predict_error)
        #self.v_previous_weights     = self.v_current_weights
        self.v_current_weights[0]   += scaled_predict_error
        self.v_current_weights[1:]  = np.add(delta_w,self.v_current_weights[1:])
        self.max_abs_error          = max(np.abs(delta_w))

    def is_converged(self):
        result = True if (self.max_abs_error < self.v_convergence_rate) else False
        return result
    
    """def calculate_abs_error(self):
        abs_error = np.abs(np.subtract(self.v_current_weights,self.v_previous_weights))
        print(abs_error)
        self.max_abs_error = max(abs_error) """       
    
    def run_preceptron(self):
        self.get_data_dimension(self.v_data_predictors)
        #bool_end = False
        iteration_step = 0
        self.v_success_cases = 0
        self.v_max_iterations = self.v_num_rows*self.v_num_passes
        while iteration_step<self.v_max_iterations:
            if iteration_step==0:
                self.initialize_weights()
            n = iteration_step%self.v_num_rows
            x = self.v_data_predictors[n]
            z = self.calculate_net_input(self.v_current_weights,self.v_data_predictors[n])
            y_predicted = self.evaluate_threshold_function(z)
            y_label = self.v_y_label[n]
            if y_predicted-y_label==0.0:
                self.v_success_cases+=1
            #print("{} w={}".format(iteration_step, self.v_current_weights))
            self.update_weights(y_label=y_label,
                                y_predicted=y_predicted,
                                x=x)            #self.calculate_abs_error()
            #bool_end = self.is_converged()
            iteration_step+=1
            #print("{} {}".format(iteration_step, self.max_abs_error))
        self.run_successfully = True

    def get_optimized_weights(self):
        if self.run_successfully:
            return self.v_current_weights
        
    def print_success_rate(self):
        print("Success rate = {}".format(self.v_success_cases/self.v_max_iterations))

    def single_predict(self, w, x):
        z = self.calculate_net_input(w,x)
        y_predicted = self.evaluate_threshold_function(z)
        return y_predicted

    def array_predict(self, w, X):
        N = np.shape(X)[0]
        Y_predicted = np.array([self.single_predict(w,X[i]) for i in range(N)])
        return Y_predicted

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
    for n in range(6):
        model = perceptron(data_predictors=stratified_training_set_predictors,
                       data_labels=stratified_training_set_labels,
                       eta=0.2,
                       convergence_rate=10**(-6),
                       threshold_value=0,
                       thresh_pass=1.0,
                       thresh_fail=-1.0,
                       initial_weights=None,
                       random_seed=1,
                       n_passes=5**n)
        model.run_preceptron()
        weights = model.get_optimized_weights()
        print("optimized weights for n_passes = {}:".format(5**n))
        print(weights)
        model.print_success_rate()
        print("\n")
        Y_predicted = model.array_predict(weights, stratified_test_set_predictors)
        error_array = np.subtract(Y_predicted, stratified_test_set_labels)
        dim_test    = np.shape(error_array)[0]
        accuracy = (dim_test - np.count_nonzero(error_array))/dim_test
        print("accuracy for n_passes = {}:".format(5**n))
        print(accuracy)
        print("\n")
        print("\n")


if __name__ == "__main__":
    main()