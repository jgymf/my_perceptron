import numpy as np
import random


class perceptron:
    def __init__(self, 
                 data_predictors,
                 data_labels,
                 eta, 
                 threshold_value,
                 thresh_pass=1.0,
                 thresh_fail=-1.0,
                 initial_weights=None,
                 random_seed=None,
                 n_passes=20):
        self.v_data_predictors  = data_predictors
        self.v_y_label          = data_labels
        self.v_eta              = eta
        self.v_threshold_value  = threshold_value
        self.v_thresh_pass      = thresh_pass
        self.v_thresh_fail      = thresh_fail
        self.v_initial_weights  = initial_weights
        self.v_random_seed      = random_seed
        self.v_num_passes       = n_passes
        self.v_net_input        = None
        self.v_max_iterations   = None
        self.accuracy           = None
        #self.max_abs_error      = None
        self.__current_weights  = None
        self.__num_rows         = None
        self.__num_columns      = None
        self.__run_successfully = False
        self.update_per_epoch = []


    def __del__(self):
        print("Perceptron object is going to be destroyed...")

    def __get_data_dimension(self,X):
        self.__num_rows         = np.shape(X)[0]
        self.__num_columns      = np.shape(X)[1]
        #print("row dimension is {}".format(self.__num_rows))
        #print("column dimension is {}".format(self.__num_columns))

    def __initialize_weights(self):
        self.__current_weights  = self.v_initial_weights
        if self.v_initial_weights is None:
            if self.v_random_seed   == None:
                self.v_random_seed  = random.randint(1,10**6)
            print("The random seed used to initialize the weight vector is {}.".format(self.v_random_seed))
            random.seed(self.v_random_seed)
            #self.__current_weights = np.random.random(self.__num_columns)
            random_draw  = np.random.RandomState(seed=self.v_random_seed)
            self.__current_weights  = random_draw.normal(loc=0.0, scale=0.1, size=1+self.__num_columns)
            #print("This is the initial weight vector:")
            #print(self.__current_weights)
        if len(self.__current_weights) != self.__num_columns + 1:
                print("ERROR: Dimension of initial weight vector does not match dimension of dataset (i.e., number of columns)\n")
                print("\n ... Exiting now.\n")
                exit
        #self.v_previous_weights = np.zeros(shape=self.__num_columns+1, dtype=float)

    def calculate_net_input(self, w, x):
        #c = np.dot(np.transpose(self.__current_weights),self.v_data_predictors[iter_num])
        c = np.sum(np.multiply(w[1:],x))
        c += w[0]
        #c = np.sum(np.multiply(self.__current_weights[1:], self.v_data_predictors[iter_num]))
        #c += self.__current_weights[0]
        return c
    
    def evaluate_threshold_function(self, z):
        return (self.v_thresh_pass if z>=self.v_threshold_value else self.v_thresh_fail)
    
    def __update_weights(self,y_label, y_predicted, x):
        scaled_predict_error        = self.v_eta*(y_label-y_predicted)
        delta_w                     = np.multiply(x, scaled_predict_error)
        #self.v_previous_weights     = self.__current_weights
        self.__current_weights[0]   += scaled_predict_error
        self.__current_weights[1:]  = np.add(delta_w,self.__current_weights[1:])
        #self.max_abs_error          = max(np.abs(delta_w))

    def __update_weights_2(self, y_label, y_predicted, x):
        s_error         = self.v_eta*(y_label-y_predicted)
        delta_w               = np.multiply(x, s_error+0.5*s_error**2)
        #delta_w         = np.multiply(np.sin(x), v)
        #delta_w         = np.add(delta_w_1, delta_w_2)
        self.__current_weights[0]   += s_error
        self.__current_weights[1:]  = np.add(delta_w,self.__current_weights[1:]) 

    def __choose_w_update_func(self, n=1):
        func = None
        if n==1:
            func = self.__update_weights
        elif n==2:
            func = self.__update_weights_2
        else:
            print("ERROR: Chosen option for weight updating function is not valid. Choose either 1 or 2.\n")
        return func
        

    def __run_perceptron(self, w_update_func):
        self.__get_data_dimension(self.v_data_predictors)
        #bool_end = False
        iteration_step = 0
        self.v_success_cases = 0
        self.v_max_iterations = self.__num_rows*self.v_num_passes
        while iteration_step<self.v_max_iterations:
            if iteration_step==0:
                self.__initialize_weights()
            n = iteration_step%self.__num_rows
            x = self.v_data_predictors[n]
            z = self.calculate_net_input(self.__current_weights,self.v_data_predictors[n])
            y_predicted = self.evaluate_threshold_function(z)
            y_label = self.v_y_label[n]
            if y_predicted-y_label==0.0:
                self.v_success_cases+=1
            #print("{} w={}".format(iteration_step, self.__current_weights))
            w_update_func(y_label=y_label,
                          y_predicted=y_predicted,
                          x=x)            #self.calculate_abs_error()
            #bool_end = self.is_converged()
            if n==self.__num_rows-1:
                self.amend_update_per_epoch_array(iteration_step)
            iteration_step+=1
            #print("{} {}".format(iteration_step, self.max_abs_error))
        self.__run_successfully = True

    def get_optimized_weights(self):
        return self.__current_weights if self.__run_successfully else -1
        
    def print_optimized_weights(self, n_decimals=4):
        if self.__run_successfully:
            print("Final weight vector is: {}".format(np.round(self.__current_weights, n_decimals)))
        else:
            print("ERROR: Perceptron not run successfully. Cannot print weights.")
        
    def amend_update_per_epoch_array(self, i):
            n_update = 0
            if len(self.update_per_epoch)>0:
                n_update = i+1- self.v_success_cases - sum(self.update_per_epoch)
            else:
                n_update = self.__num_rows- self.v_success_cases
            self.update_per_epoch.append(n_update)        

    def print_success_rate(self, n_decimals=4):
        if self.__run_successfully:
            print("Success rate = {}".format(round(self.v_success_cases/self.v_max_iterations, n_decimals)))
        else:
            print("ERROR: Perceptron not run successfully. Cannot print success rate.")

    def fit(self, w_update_method=1, n_digits=4):
        w_chosen_func = self.__choose_w_update_func(w_update_method)
        self.__run_perceptron(w_update_func=w_chosen_func)
        self.print_optimized_weights(n_decimals=n_digits)
        self.print_success_rate(n_decimals=n_digits)
        return self

    def single_predict(self, w, x):
        z = self.calculate_net_input(w,x)
        y_predicted = self.evaluate_threshold_function(z)
        return y_predicted

    def array_predict(self, w, X):
        N = np.shape(X)[0]
        Y_predicted = np.array([self.single_predict(w,X[i]) for i in range(N)])
        return Y_predicted
    
    def calculate_accuracy(self, Y_label, Y_predicted):
        error_array = np.subtract(Y_label, Y_predicted)
        dim_test    = np.shape(error_array)[0]
        self.accuracy = (dim_test - np.count_nonzero(error_array))/dim_test
        return self.accuracy
    
    def get_accuracy(self):
        return self.accuracy
    
    def get_update_per_epoch(self, n_decimals=4):
        return (np.round(np.array(self.update_per_epoch), n_decimals))

    def fit_and_print_accuracy(self, X_test, Y_test, w_update_method=1, n_digits=4):
        self.fit(w_update_method=w_update_method, n_digits=n_digits)
        weight_vector = self.get_optimized_weights()
        Y_predicted = self.array_predict(weight_vector, X_test)
        accuracy = self.calculate_accuracy(Y_test, Y_predicted)
        print("Accuracy = {}".format(round(accuracy,n_digits)))