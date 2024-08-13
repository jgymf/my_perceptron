import sys
import os
import numpy as np
from perceptron import perceptron

class adaline(perceptron):
    def __init__(self,
                 data_predictors,
                 data_labels,
                 learning_rate=10**(-3), 
                 threshold_value=0,
                 thresh_pass=1.0,
                 thresh_fail=-1.0,
                 initial_weights=None,
                 random_seed=None,
                 n_epochs=20):
        super().__init__( data_predictors,
                         data_labels,
                         learning_rate=10**(-3),
                         threshold_value=0,
                         thresh_pass=1.0,
                         thresh_fail=-1.0,
                         initial_weights=None,
                         random_seed=None,
                         n_epochs=20)
        self.transpose_data_predictors = np.transpose(data_predictors)
        self.success_per_epoch = []
        self.SSE_per_epoch = []
    """
    fUNCTIONS TO BE UNINHERITED:
    """        
    get_update_per_epoch            = property(doc='(!) method not inherited')
    array_predict                   = property(doc='(!) method not inherited')
    single_predict                  = property(doc='(!) method not inherited')
    amend_update_per_epoch_array    = property(doc='(!) method not inherited')
    __run_perceptron                = property(doc='(!) method not inherited')
    __run_perceptron_iter           = property(doc='(!) method not inherited')
    __update_weights_3              = property(doc='(!) method not inherited')
    __update_weights_2              = property(doc='(!) method not inherited')


    def calculate_net_input(self, W, X_t):
        return np.add(np.matmul(W[1:],X_t),W[0])
        
    def evaluate_activation_function(self, A):
        return A
        
    def evaluate_activation_function_2(self, A):
        return np.tanh(A)

    def evaluate_threshold_function(self, Z):
        return np.where(Z>=self.threshold_value, self.thresh_pass, self.thresh_fail)

    def choose_w_n_act_func(self, n=1):
        """
        Objective:  
        ---------
                    Decide which weight updating rule to use.

        Parameters:
        ----------
                    n   : an integer representing a weight updating rule.

        Returns:
        -------
                    a weight updating function.        
        """ 
        w_func      = None
        act_func    = None
        if n==1:
            w_func      = self.__update_weights
            act_func    = self.evaluate_activation_function
        elif n==2:
            w_func      = self.__update_weights_2
            act_func    = self.evaluate_activation_function_2
        #elif n==3:
        #    w_func = self.__update_weights_3
        else:
            print("ERROR: Chosen option for weight updating function is not valid. Choose either 1, 2 or 3.\n")
            exit
        return w_func, act_func

    def __update_weights(self,error_vector):
        delta_W                     = np.dot(error_vector, self.data_predictors)*self.learning_rate
        self.current_weights[0]     += self.learning_rate*np.sum(error_vector)
        self.current_weights[1:]    += delta_W

    def __update_weights_2(self,error_vector):
        #error_vector                = np.subtract(self.Y_label,Y_predicted)
        error_vector_sech2          = np.multiply(error_vector, 1.0/(np.cosh(self.Y_label-error_vector))**2)
        delta_W                     = np.dot(error_vector_sech2, self.data_predictors)*self.learning_rate
        self.current_weights[0]     += self.learning_rate*np.sum(error_vector_sech2)
        self.current_weights[1:]    += delta_W

    def __update_SSE_per_epoch(self, error_vector):
        self.SSE_per_epoch.append(np.sum(error_vector**2)/2.0)

    def __run_adaline_iter(self, iteration_step, w_update_func, act_func):
        if iteration_step==0:
            super().initialize_weights()
            #reshape Y_label and current_weights for matrix product consistency
            np.reshape(self.Y_label, (1, len(self.Y_label)))
            np.reshape(self.current_weights, (1, len(self.current_weights)))
        #n = iteration_step%self.num_rows
        #x = self.data_predictors[n]
        net_input_vector     = self.calculate_net_input(self.current_weights,self.transpose_data_predictors)
        np.reshape(net_input_vector, (1,self.num_rows))
        #print("shape of net_input_vector = ", np.shape(net_input_vector))
        Y_predicted          = act_func(A=net_input_vector)
        #print("Y_predicted = ", Y_predicted)
        #print("Y_predicted shape = ", np.shape(Y_predicted))
        #print("Y_label = ", self.Y_label)
        #print("Y_label shape = ", np.shape(self.Y_label))
        error_vector         = np.subtract(self.Y_label, Y_predicted)
        #print("error_vector = ", error_vector)
        num_false_predicts   = np.count_nonzero(np.subtract(self.Y_label,Y_predicted))
        self.success_cases += self.num_rows - num_false_predicts
        self.__update_SSE_per_epoch(error_vector=error_vector)
        w_update_func(error_vector=error_vector)            
        self.amend_success_per_epoch_array() 

    def __run_adaline(self, w_update_func, act_func):
        super().get_data_dimension(self.data_predictors)
        self.success_cases          = 0
        self.max_iterations         = self.n_epochs
        dummy = [self.__run_adaline_iter(iteration_step=iteration_step,
                                         w_update_func=w_update_func, 
                                         act_func=act_func) for iteration_step in range(self.max_iterations)]
        if len(dummy)==self.max_iterations:
            self.run_successfully = True

    def fit(self, w_update_method=1, n_digits=4):
        print("Running adaline ...\n")
        w_chosen_func, activation_func = self.choose_w_n_act_func(w_update_method)
        self.__run_adaline(w_update_func=w_chosen_func, act_func=activation_func)
        self.print_optimized_weights(n_decimals=n_digits)
        self.print_success_rate(n_decimals=n_digits)        

    def amend_success_per_epoch_array(self):
        self.success_per_epoch.append(self.success_cases)

    def get_success_per_epoch(self):
        return self.success_per_epoch
    
    def get_SSE_per_epoch(self):
        return self.SSE_per_epoch
    
    def get_update_per_epoch(self):
        return np.subtract(self.num_rows,self.success_per_epoch)
    
    def fit_and_print_accuracy(self, X_test, Y_test, w_update_method=1, n_digits=4):
        """
        Objective:  
        ---------
                    Run perceptron algorithm to get the optimized weights, use the latter to
                    validate the model by running it to predict the traget values on a test dataset,
                    and print the accuracy obtained at the validation step.

        Parameters:
        ----------
                    X_test          : a numpy array of predictive features to test the model on.
                    Y_test          : a numpy array of the true target values of X_test.
                    w_update_method : an integer representing which weight update rule to use.
                                      Default value is 1.
                    n_digits        : decimal precision at which to print any result to the console.

        Returns:
        -------
                    (implicit) None
        """ 
        self.fit(w_update_method=w_update_method, n_digits=n_digits)
        weight_vector   = self.get_optimized_weights()
        Z               = self.calculate_net_input(weight_vector, np.transpose(X_test))
        Z               = self.evaluate_activation_function(Z)
        Y_predicted     = self.evaluate_threshold_function(Z)
        accuracy        = self.calculate_accuracy(Y_test, Y_predicted)
        m_string        = '{0: <{width}}'.format("Accuracy", width=self.width)
        print(m_string + ":", round(accuracy,n_digits))
        print("\n")