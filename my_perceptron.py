import numpy as np
import random


class perceptron:
    def __init__(self, 
                 data_predictors,
                 data_labels,
                 initial_weights,
                 eta, 
                 convergence_rate,
                 threshold_value,
                 thresh_pass=1.0,
                 thresh_fail=-1.0,
                 random_seed=None,
                 max_iterations=1000):
        self.v_data_predictors  = data_predictors
        self.v_current_weights  = initial_weights
        self.v_previous_weights = None
        self.v_X                = None
        self.v_y_label          = data_labels
        self.v_y_predicted      = None
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
        self.v_max_iterations   = max_iterations
        self.max_abs_error      = None

    def get_data_dimension(self):
        self.v_num_rows         = np.shape(self.v_data)[0]
        self.v_num_columns      = np.shape(self.v_data)[1]

    def initialize_weights(self):
        if self.v_initial_weights is None:
            if self.v_random_seed == None:
                self.v_random_seed = random.randint(1,10000)
            print("The random seed used to initialize the weight vector is {}\n".format(self.v_random_seed))
            random.seed(self.v_random_seed)
            self.v_current_weights = random.random_sample(self.v_num_columns)
        else:
            if len(self.v_initial_weights) != self.v_num_columns:
                print("ERROR: Dimension of initial weight vector does not match dimension of dataset (i.e., number of columns)\n")
                print("\n ... Exiting now.\n")
                exit

    def calculate_net_input(self, iter_num):
        c = np.dot(np.transpose(self.v_current_weights),self.v_data_predictors[iter_num])
        return c
    
    def evaluate_threshold_function(self, z):
        phi = self.v_thresh_fail
        if z >= self.v_threshold_value:
            phi = self.v_thresh_pass
        return phi
    
    def update_weights(self,iter_num):
        scaled_predict_error = self.eta*(self.v_y_label[iter_num]-self.v_y_predicted)
        delta_w = np.multiply(self.v_data_predictors[iter_num], scaled_predict_error)
        self.v_previous_weights = self.v_current_weights
        self.v_current_weights = (np.add(delta_w+self.v_current_weights))

    def is_converged(self):
        abs_error = np.abs(np.subtract(self.v_current_weights,self.v_previous_weights))
        self.max_abs_error = max(abs_error)
        result = True if (self.max_abs_error < self.v_convergence_rate) else False
        return result
    
    def run_preceptron(self):
        bool_end = False
        iteration_step = 0
        while bool_end == False and iteration_step<self.v_max_iterations:
            if iteration_step==0:
                self.initialize_weights()
            z = self.evaluate_threshold_function(iteration_step)
            self.v_y_predicted = self.evaluate_threshold_function(z)
            self.update_weights(iteration_step)
            bool_end = self.is_converged()
            iteration_step+=1
            print("{} {}".format(iteration_step, self.max_abs_error))