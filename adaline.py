import sys
import os
import numpy as np
from perceptron import perceptron

class adaline(perceptron):
    """
        This is a Binary Calssifier.
        
        Parameters:
        ----------
        *data_predictors    = a numpy array of predictors to used for training the perceptron.
                              The columns of the array represent the features, while the rows represent different instances of the data.
        *data_labels        = a numpy array of labels to be used for training the perceptron
        *learning_rate      = rate at which the algorithm learns. Default value is 10^(-3).
        *threshold_value    = boundary value in the step function. Default value is 0.
        *thresh_pass        = one of the possible values the step function can assume. Default value is 1.
        *thresh_fail        = the other possible value the step function can have. Default value is -1.
        *initial_weights    = a numpy 1D array of weights the user would like the algorithm to start with.
                              If not provided, the algorithm creates one with random entries, according to a normal distribution centered on 0.
                              Note that the dimension of the this numpy array must be = 1 + n_C, where n_C is the number of columns.
        *random_seed        = Seed for the random generation of the initial_weights vector.
        *n_epochs           = number of times the algorithm is supposed to repeat training on the provided training dataset. Default value is 20.
    """
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
        self._transpose_data_predictors = np.transpose(data_predictors)
        self._success_per_epoch = []
        self._SSE_per_epoch = []
    """
    FUNCTIONS TO BE UNINHERITED:
    """        
    array_predict                   = property(doc='(!) method not inherited')
    single_predict                  = property(doc='(!) method not inherited')
    amend_update_per_epoch_array    = property(doc='(!) method not inherited')


    def calculate_net_input(self, W, X_t):
        return np.add(np.matmul(W[1:],X_t),W[0])
        
    def evaluate_activation_function(self, A):
        """
        Activation function for method_1.
        """
        return A
        
    def evaluate_activation_function_2(self, A):
        """
        Activation function for method_2.
        """
        return np.tanh(A)

    def evaluate_threshold_function(self, Z):
        return np.where(Z>=self.threshold_value, self.thresh_pass, self.thresh_fail)

    def choose_w_n_act_func(self, n=1):
        """
        Objective:  
        ---------
                    Decide which weight updating rule (w_func) and which activation function (act_func)
                    to use.

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
        else:
            print("ERROR: Chosen option for weight updating function is not valid. Choose either 1, 2 or 3.\n")
            exit
        return w_func, act_func

    def __update_weights(self,error_vector):
        """
        Weights update function for method_1.
        """
        delta_W                     = np.dot(error_vector, self.data_predictors)*self.learning_rate
        self._current_weights[0]     += self.learning_rate*np.sum(error_vector)
        self._current_weights[1:]    += delta_W

    def __update_weights_2(self,error_vector):
        """
        Weights update function for method_2.
        """
        error_vector_sech2          = np.multiply(error_vector, 1.0/(np.cosh(self.Y_label-error_vector))**2)
        delta_W                     = np.dot(error_vector_sech2, self.data_predictors)*self.learning_rate
        self._current_weights[0]     += self.learning_rate*np.sum(error_vector_sech2)
        self._current_weights[1:]    += delta_W

    def __update_SSE_per_epoch(self, error_vector):
        self._SSE_per_epoch.append(np.sum(error_vector**2)/2.0)

    def __run_adaline_iter(self, iteration_step, w_update_func, act_func):
        if iteration_step==0:
            super().initialize_weights()
            np.reshape(self.Y_label, (1, len(self.Y_label)))
            np.reshape(self._current_weights, (1, len(self._current_weights)))
        net_input_vector     = self.calculate_net_input(self._current_weights,self._transpose_data_predictors)
        np.reshape(net_input_vector, (1,self._num_rows))
        Y_predicted          = act_func(A=net_input_vector)
        error_vector         = np.subtract(self.Y_label, Y_predicted)
        num_false_predicts   = np.count_nonzero(np.subtract(self.Y_label,Y_predicted))
        self._success_cases += self._num_rows - num_false_predicts
        self.__update_SSE_per_epoch(error_vector=error_vector)
        w_update_func(error_vector=error_vector)            
        self.amend_success_per_epoch_array() 

    def __run_adaline(self, w_update_func, act_func):
        super().get_data_dimension(self.data_predictors)
        self._success_cases          = 0
        self._max_iterations        = self.n_epochs
        dummy = [self.__run_adaline_iter(iteration_step=iteration_step,
                                         w_update_func=w_update_func, 
                                         act_func=act_func) for iteration_step in range(self._max_iterations)]
        if len(dummy)==self._max_iterations:
            self._run_successfully  = True
            self._final_weights     = self._current_weights

    def fit(self, w_update_method=1, n_digits=4):
        print("Running adaline ...\n")
        w_chosen_func, activation_func = self.choose_w_n_act_func(w_update_method)
        self.__run_adaline(w_update_func=w_chosen_func, act_func=activation_func)
        self.print_optimized_weights(n_decimals=n_digits)
        self.print_success_rate(n_decimals=n_digits)
        self._update_per_epoch = np.subtract(self._num_rows,self._success_per_epoch)      

    def amend_success_per_epoch_array(self):
        self._success_per_epoch.append(self._success_cases)
    
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
        weight_vector   = self._final_weights
        Z               = self.calculate_net_input(weight_vector, np.transpose(X_test))
        Z               = self.evaluate_activation_function(Z)
        Y_predicted     = self.evaluate_threshold_function(Z)
        accuracy        = self.calculate_accuracy(Y_test, Y_predicted)
        m_string        = '{0: <{width}}'.format("Accuracy", width=self.width)
        print(m_string + ":", round(accuracy,n_digits))
        print("\n")
