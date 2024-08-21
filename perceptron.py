import numpy as np
import random


class IncompleteRunError(Exception):
    pass

class perceptron:
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
        """
        This is a Perceptron Binary Calssifier.
        
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

        Attributes:
        ----------
        The parameters mentioned are all also attributes of the class, maintaining the same parameter nomenclature. The exception is: 
        *Y_label, which is just the equivalent of data_labels parameter mentioned above.

        The remaining attributes are:
        *max_iterations     = n_R * n_epochs, where n_R is the number of rows in data_predictors
        *accuracy           = (number of predictions made right)/(number of test instances)
        *update_per_epoch   = a list where the n-th entry gives the number of times the weight vector was updated during the n-th epoch run
        
        """
        self.data_predictors    = data_predictors
        self.Y_label            = data_labels
        self.learning_rate      = learning_rate
        self.threshold_value    = threshold_value
        self.thresh_pass        = thresh_pass
        self.thresh_fail        = thresh_fail
        self._initial_weights    = initial_weights
        self.random_seed        = random_seed
        self.n_epochs           = n_epochs
        self._max_iterations     = None
        self._update_per_epoch   = []
        self._SSE_per_epoch      = []
        self._accuracy           = None
        self._final_weights      = None
        self._current_weights    = None
        self._num_rows           = None
        self._num_columns        = None
        self._run_successfully   = False
        self._success_cases      = None
        self._SSE_accumulated    = None
        self.width              = 40


    def get_data_dimension(self,X):
        """
        Objective:
        --------- 
                    Private method to get shape of numpy array X, and store shape as private variables.

        Parameters:
        ----------
                    *X: a numpy array

        Returns:
        -------
                    (implicit) None
        """
        self._num_rows         = np.shape(X)[0]
        self._num_columns      = np.shape(X)[1]


    def initialize_weights(self):
        """
        Objective:  
        ---------
                    Private method to initialize the weights. 

        Parameters:
        ----------
                    None

        Returns:
        -------
                    (implicit) None

        How it works:
        ------------
                    If user doesn't provide an initial weights (1D numpy array), a random seed is need to generate the weights vector.
                    User may choose to provide a random seed at the instantiation of the class.
                    If a random seed is not provided, the method generates one.
                    Beacuase of the bias unit, the dimension of the weight vector must be a unit greater than 
                    the number of features in the dataset (i.e., number of columns).
                    The method checks to see if that condition is satisfied. It exits if the condition is not satisfied.
        
        """
        self._current_weights  = self._initial_weights
        if self._initial_weights is None:
            if self.random_seed   == None:
                self.random_seed  = random.randint(1,10**6)
            print("The random seed used to initialize the weight vector is {}.".format(self.random_seed))
            random.seed(self.random_seed)
            random_draw  = np.random.RandomState(seed=self.random_seed)
            self._current_weights  = random_draw.normal(loc=0.0, scale=0.1, size=1+self._num_columns)
        if len(self._current_weights) != self._num_columns + 1:
                print("ERROR: Dimension of initial weight vector does not match dimension of dataset (i.e., number of columns)\n")
                print("\n ... Exiting now.\n")
                exit


    def calculate_net_input(self, w, x):
        """
        Objective:  
        ---------
                    Calculate the net input. 

        Parameters:
        ----------
                    w: a numpy 1D array, representing the (current) weights.
                    x: a numpy 1D array (representing a row of the predictor array)

        Returns:
        -------
                    c: a float 

        How it works:
        ------------
                    The method does a weighted sum of x using the array w as weights.
        
        """
        c = np.sum(np.multiply(w[1:],x))
        c += w[0]
        return c


    def evaluate_threshold_function(self, z):
        """
        Objective:  
        ---------
                    Determine output value of the step-function. 

        Parameters:
        ----------
                    z: a float representing the net input.

        Returns:
        -------
                    a float (either thresh_pass or thresh_fail)
        
        """       
        return (self.thresh_pass if z>=self.threshold_value else self.thresh_fail)


    def __update_weights(self,y_label, y_predicted, x):
        """
        Objective:  
        ---------
                    Update the weight vector (Method 1).

        Parameters:
        ----------
                    y_label     : a float representing the expected value of the target feature.
                    y_predicted : a float representing the predicted value of the target feature.
                    X           : a numpy 1D array representing predictive features used to calculate y_predicted.

        Returns:
        -------
                    (implicit) None

        How it works:
        ------------
                    This is just an implementation of the Rosenblatt method, whereby the i-th change in w, the weight vector,
                    is performed according to the rule:
                    .. math::
                        \Delta w_i = \eta (y^{label}_i-y^{predicted}_i) X_i
                    where X_i is the i-th array of predictive features, and eta is the learning rate.
        
        """       
        scaled_predict_error        = self.learning_rate*(y_label-y_predicted)
        delta_w                     = np.multiply(x, scaled_predict_error)
        self._current_weights[0]   += scaled_predict_error
        self._current_weights[1:]  = np.add(delta_w,self._current_weights[1:])


    def __update_weights_2(self, y_label, y_predicted, x):
        """
        Objective:  
        ---------
                    Update the weight vector (Method 1).

        Parameters:
        ----------
                    y_label     : a float representing the expected value of the target feature.
                    y_predicted : a float representing the predicted value of the target feature.
                    X           : a numpy 1D array representing predictive features used to calculate y_predicted.

        Returns:
        -------
                    (implicit) None

        How it works:
        ------------
                    A modification of the Rosenblatt method. The i-th change in w, the weight vector,
                    is performed according to the rule:
                    .. math::
                        \Delta w_i = \eta (y^{label}_i-y^{predicted}_i) *[1 + 0.5*\eta (y^{label}_i-y^{predicted}_i)] X_i
                    where X_i is the i-th array of predictive features, and eta is the learning rate.
        
        """ 
        s_error                     = self.learning_rate*(y_label-y_predicted)
        delta_w                     = np.multiply(x, s_error+0.5*s_error**2)
        self._current_weights[0]   += s_error
        self._current_weights[1:]  = np.add(delta_w,self._current_weights[1:])


    def __update_weights_3(self, y_label, y_predicted, x):
        """
        Objective:  
        ---------
                    Update the weight vector (Method 1).

        Parameters:
        ----------
                    y_label     : a float representing the expected value of the target feature.
                    y_predicted : a float representing the predicted value of the target feature.
                    X           : a numpy 1D array representing predictive features used to calculate y_predicted.

        Returns:
        -------
                    (implicit) None

        How it works:
        ------------
                    The i-th change in w, the weight vector, is performed according to the rule:
                    .. math::
                        \Delta w_i = tanh[\eta (y^{label}_i-y^{predicted}_i) * X_i]
                    where X_i is the i-th array of predictive features, and eta is the learning rate.
        
        """ 
        s_error                     = self.learning_rate*(y_label-y_predicted)
        r                           = np.multiply(x, s_error)
        delta_w                     = np.tanh(r)
        self._current_weights[0]    += s_error
        self._current_weights[1:]   = np.add(delta_w,self._current_weights[1:])


    def choose_w_update_func(self, n=1):
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
        func = None
        if n==1:
            func = self.__update_weights
        elif n==2:
            func = self.__update_weights_2
        elif n==3:
            func = self.__update_weights_3
        else:
            print("ERROR: Chosen option for weight updating function is not valid. Choose either 1, 2 or 3.\n")
            exit
        return func
        

    def __run_perceptron_iter(self, iteration_step, w_update_func):
        """
        Objective:  
        ---------
                    Run the perceptron algorithm on a 1D array of predictive features.
                    Also updates:
                      i) the weight vector and 
                      ii) if the 1D array represents the end of training dataset,
                          the number of times the weight vector has been updated is calculated and stored to a list.

        Parameters:
        ----------
                    iteration step  : a float representing the expected value of the target feature.
                    w_update_func   : a float representing the predicted value of the target feature.

        Returns:
        -------
                    (implicit) None

        How it works:
        ------------
                    The iteration_step parameter represents the j-th time take a 1D array of predictive features to train on.
                    For example, if the dimension of the training set is 500 entries, and the number of epochs we want to run
                    the training is 10, then 0 <= iteration_step <= (500*10-1).

                    The iteration_parameter is first converted into a integer n,
                    where n = (iteration_step mod dimension_of_training_dataset).
                    n represents an index of 1D array of predictive features in the training dataset.
                    In other words, n refers to a row number in the training set.

                    The method then picks the n-th row of the training set as x, calculates the net input based on the values of x
                    and the current weights. It runs the obtained net input value through the step function and obtains a prediction,
                    y_predicted, of the target binary feature.

                    Once y_predicted is in hand, it fetches the corresponding true value, y_label, according to the training dataset
                    and evalues the error. If the prediction was right, it increments the number of times it has been right by a unit.
                    In addition, with y_label, y_predicted and x, the method calls a weight updating rule of the user's choice
                    to update the weights.

                    Finally, if the n-th 1D array of predictive features happen to be the last row of our training set, the method calls
                    another function to calculate and store to a list how many times the weight vector has been changed.
        """ 
        if iteration_step==0:
            self.initialize_weights()
        n = iteration_step%self._num_rows
        x = self.data_predictors[n]
        z = self.calculate_net_input(self._current_weights,self.data_predictors[n])
        y_predicted             = self.evaluate_threshold_function(z)
        y_label                 = self.Y_label[n]
        error                   = y_predicted-y_label
        self._SSE_accumulated    += error**2 
        if error    == 0.0:
            self._success_cases+=1
        w_update_func(y_label=y_label,
                      y_predicted=y_predicted,
                      x=x)            
        if n==self._num_rows-1:
            # calculate and store the number of times the weights were update
            self.amend_update_per_epoch_array(iteration_step)
            self.amend_SSE_per_epoch_array(iteration_step)


    def __run_perceptron(self, w_update_func):
        """
        Objective:  
        ---------
                    Run the perceptron algorithm on the full training dataset, n_epochs of time.

        Parameters:
        ----------
                    w_update_func   : a function implementing a weight updating rule.

        Returns:
        -------
                    (implicit) None

        How it works:
        ------------
                    The total number of instances, refered to as "max_iterations" in the method, we have at our disposal
                    to update the weight vector is given by the product of the number of epochs (n_epochs) 
                    and the number of rows in the training dataset.

                    The method calls the function "__run_perceptron_iter" to run the perceptron algorithm for a total of
                    max_iterations of times through a list comprehension call. This probably increases the efficiency of the program
                    when max_iterations is pretty high.

                    If a total of max_iterations of the function "__run_perceptron_iter" is successfully run, then the private attribute
                    "__run_successfully" is set to True. This attribute controls several other methods of the class, especially
                    the printing methods.

        """ 
        self.get_data_dimension(self.data_predictors)
        iteration_step          = 0
        self._success_cases      = 0
        self._SSE_accumulated    = 0
        self._max_iterations = self._num_rows*self.n_epochs
        dummy = [self.__run_perceptron_iter(iteration_step, w_update_func) for iteration_step in range(self._max_iterations)]
        if len(dummy)==self._max_iterations:
            self._run_successfully  = True
            self._final_weights     = self._current_weights

        
    def print_optimized_weights(self, n_decimals=4):
        """
        Objective:  
        ---------
                    Print to console the final weight vector a successful run of the "__run_perceptron" function.
                    A string containing the final weights (if "__run_perceptron" was successfully run), 
                    else you get an error message.

        Parameters:
        ----------
                    n_decimals  : fixed number of decimals to use when printing the entries of the weight vector.
                                   Default value is 4.

        Returns:
        -------
                    (implicit) None
        """ 
        if self._run_successfully:
            m_string = '{0: <{width}}'.format("Final weight vector", width=self.width)
            print(m_string + ":", np.round(self._current_weights, n_decimals))
        else:
            print("ERROR: Perceptron not run successfully. Cannot print weights.")

    
    def amend_SSE_per_epoch_array(self, i):
        new_SSE = 0
        if len(self._SSE_per_epoch)>0:
            new_SSE = self._SSE_accumulated - sum(self._SSE_per_epoch)
        else:
            new_SSE = self._SSE_accumulated
        self._SSE_per_epoch.append(new_SSE)

    def amend_update_per_epoch_array(self, i):
        """
        Objective:  
        ---------
                    Update the list "update_per_epoch", i.e. a list whose entries record how many times
                    the weight vector has been updated during each epoch run of the perceptron algorithm.

        Parameters:
        ----------
                    i  : an integer representing the i-th time we have taken a 1D array of predictive features
                         from the training dataset to train the model.

        Returns:
        -------
                    (implicit) None
        """ 
        n_update        = 0
        if len(self._update_per_epoch)>0:
            n_update    = i+1- self._success_cases - sum(self._update_per_epoch)
        else:
            n_update    = self._num_rows- self._success_cases
        self._update_per_epoch.append(n_update)


    def print_success_rate(self, n_decimals=4):
        """
        Objective:  
        ---------
                    Print to console the __success_rate attribute.
                    The __success_rate is defined as the ratio between the number of times the algorithm
                    rightly predicted the target and the total number of attempts. 
                    The method prints a string containing the __success_rate (if "__run_perceptron" was successfully run), 
                    else you get an error message.

        Parameters:
        ----------
                    n_decimals  : fixed number of decimals to use when printing the __success_rate attribute.
                                   Default value is 4.

        Returns:
        -------
                    (implicit) None
        """ 
        if self._run_successfully:
            m_string = '{0: <{width}}'.format("Success rate during training", width=self.width)
            print(m_string + ":", round(self._success_cases/self._max_iterations, n_decimals))
        else:
            print("ERROR: Perceptron not run successfully. Cannot print success rate.")


    def fit(self, w_update_method=1, n_digits=4):
        """
        Objective:  
        ---------
                    Run the perceptron algorithm on the provided training set, print the final weights 
                    and the __success_rate during training. 

        Parameters:
        ----------
                    w_update_method : a function implementing a weight updating rule.
                    n_digits        : fixed number of decimals to use when printingthe weights 
                                      and the __success_rate attribute. Default value is 4.

        Returns:
        -------
                    (implicit) None
        """ 
        print("Running perceptron ...\n")
        w_chosen_func = self.choose_w_update_func(w_update_method)
        self.__run_perceptron(w_update_func=w_chosen_func)
        self.print_optimized_weights(n_decimals=n_digits)
        self.print_success_rate(n_decimals=n_digits)


    def single_predict(self, w, x):
        """
        Objective:  
        ---------
                    Predict binary target feature. 

        Parameters:
        ----------
                    w   : a weight vector (with a bias unit).
                    x   : a 1D array of predictive features.

        Returns:
        -------
                    y_predicted : a binary value for the target feature.
        """ 
        z           = self.calculate_net_input(w,x)
        y_predicted = self.evaluate_threshold_function(z)
        return y_predicted


    def array_predict(self, w, X):
        """
        Objective:  
        ---------
                    Predict binary target feature for a collection of predictive features. 

        Parameters:
        ----------
                    w   : a weight vector (with a bias unit).
                    X   : an R x C array of predictive features, where R=number of rows, C=number of columns.

        Returns:
        -------
                    Y_predicted : a 1D array (of dimension R) of binary values for the target feature.
        """ 
        N           = np.shape(X)[0]
        Y_predicted = np.array([self.single_predict(w,X[i]) for i in range(N)])
        return Y_predicted


    def calculate_accuracy(self, Y_label, Y_predicted):
        """
        Objective:  
        ---------
                    Calculate accuracy of model to predict target features as ratio between
                    i) the number of times the predicted value coincided with the true value, and
                    ii) the number of test cases.

        Parameters:
        ----------
                    Y_label     : a 1D array of true values for target feature. 
                    Y_predicted : a 1D array of predicted values for target feature.

        Returns:
        -------
                    accuracy    : a float between 0 and 1 (extrema included).
        """ 
        error_array     = np.subtract(Y_label, Y_predicted)
        dim_test        = np.shape(error_array)[0]
        self._accuracy   = (dim_test - np.count_nonzero(error_array))/dim_test
        return self._accuracy
    

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
        Y_predicted     = self.array_predict(weight_vector, X_test)
        accuracy        = self.calculate_accuracy(Y_test, Y_predicted)
        m_string        = '{0: <{width}}'.format("Accuracy", width=self.width)
        print(m_string + ":", round(accuracy,n_digits))
        print("\n")

    @property
    def initial_weights(self):
        return self._initial_weights
    
    @initial_weights.setter
    def initial_weights(self, A_np_array):
        self._initial_weights = A_np_array
    
    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def update_per_epoch(self):
        return np.array(self._update_per_epoch)
    
    @property
    def SSE_per_epoch(self):
        return self._SSE_per_epoch
    
    @property
    def accuracy(self):
        if self._run_successfully is True:
            return self._accuracy
        else:
            raise IncompleteRunError("Cannot print accuracy. Perceptron did not run successfully.")
    
    @property
    def final_weights(self):
        return self._final_weights
    
    @property
    def current_weights(self):
        return self._current_weights
    
    @property
    def num_rows(self):
        return self._num_rows
    
    @property
    def num_columns(self):
        return self._num_columns
    
    @property
    def run_successfully(self):
        return self._run_successfully
    
    def success_cases(self):
        return self._success_cases
    
    def SSE_accumulated(self):
        return self._SSE_accumulated*0.50