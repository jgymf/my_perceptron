# neural_classifier_models

Here is an implementation of classic neuron classifier models like the Rosenblatt perceptron learning rule and the adaptive linear neuron (Adaline).

## Notation
Given a matrix $A$ of dimension $R \times C$, we shall represent the $n$-th row of $A$ as $A_{n\bullet}$, and the $m$-th column as $A_{\bullet m}$. 

## Perceptron
### Theory
Let $X$ be a $N \times M$ real-valued matrix of predictive features, and $Y$, a $N \times 1$ real-valued matrix of target (binary, categorical) labels. Each row of $X$, $X_{i\bullet}$, is a training sample; so there are $N$ training samples in total. Each column of $X$ represents a predictive feature (so we have $M$ predictive features in total). Without loss of generality, in the following, we shall assume we are using all $M$ features to train our perceptron.

The basic assumptions of the Rosenblatt perceptron learning rule are as follows: 
1. there is a step-function, $f(z)$ -- where $z \in \mathbb{R}$ represents a signal -- which can model the target binary feature $Y$;
2. each row of $X$ has its own signal: let $z(i)$ be the signal for $X_{i\bullet}$
3. the relationship between the signal $z(i)$ and the predictive row $X_{i\bullet}$ is linear; i.e.,
```math
z(i) = w_0 + X_{i\bullet} \cdot W \ , \qquad i \in \{1, 2, \ldots, N\}
```
where $W$ is a $M \times 1$ matrix of reals, and $w_0$ is a constant scalar. The objective of the perceptron learning rule is to determine $W$ and $w_0$.

The step-function $f(z)$ has the expression,
```math
f(z) =   \begin{cases}
b_{+1}, \ \text{if } z \geq c_{threshold}\\
b_{-1}, \ \text{otherwise}
\end{cases}
```
where $b_{+1}$ and $b_{-1}$ represent the binary values of the target feature, and $c_{threshold}$ is a constant. In many applications of this learning rule, $b_{\pm 1} = \pm1$, and $c_{threshold} = 0$.

The signal $z$, the vector $W$ and the constant $w_0$ are usually referred to as the "input value", "weight" and "bias unit", respectively.

### Implementation

Let $z(i)$, $w_0(i)$, and $W(i)$ indicate the input value, the bias unit and the weight vector relating to the $i$-th training sample, that is, $X_i$. 

The implementation of the Rosenblatt perceptron learning rule can be summarised as follows:
1. Initialize  the weight vector, $W$, and bias unit, $w_0$, with random numbers. Let's indicate these as $W(0)$ and $w_0(0)$, respectively;
2. For the $i$-th sample, compute the input value, $z(i)$, as: $z(i) = w_0(i) + X_{i \bullet} \cdot W(i)$
3. Put $z(i)$ into the step-function $f$ to predict the corresponding target value, $\tilde{Y}_i$, as: $\tilde{Y}_i = f(z(i))$;
4. Compute the prediction error on the $i$-th sample as $\epsilon(i) = Y_i - \tilde{Y}_i$;
5. Compute the bias unit and weight vector for the next sample, $X_{i+1}$, as follows:
     1) If $\epsilon(i) = 0$, then $w_0(i+1) = w_0(i) \, \text{and } \  W(i+1) = W(i)$
     2) If $\epsilon(i) \neq 0$, then $w_0(i+1) = \sum_{k \leq i}w_0(k)$ and $W(i+1) = W(i) + \eta\cdot \epsilon(i) \cdot X^T_{i\bullet}$, where $\eta$ is the learning rate (usually, $0 \leq \eta \leq 1$) and $X^T_{i\bullet}$ is the transpose of $X_{i\bullet}$;
6. Repeat steps 2 to 5 until a halting condition is met.

Usually, one runs steps 2 to 5 sequentially through the $N$ training samples and then repeat the process another $\nu-1$ number of times. The positive integer $\nu$ is commonly referred to as "epochs". Increasing $\nu$ usually leads to better estimates of the weight vector $W$ and the bias unit $w_0$.

In this implementation, the halting condition in step 6 is if the learning has been repeated on the training set $X$ for a given $\nu$ times.

### The difference between this implementation and others that have come before it
The perceptron learning rule is implemented here as a class (see the file "perceptron.py"). The class offers three different updating rules for the bias unit $w_0$ and the weight vector $W$, and this is what makes this implementation different. The user can choose which method to use. They are as follows:
#### Method 1:
The updating rule is the original by Rosenblatt, i.e., the equations in step 5 above.
#### Method 2:
The updating rules are as follows:
  1. If $\epsilon(i) = 0$, then $w_0(i+1) = w_0(i) \, \text{and } \  W(i+1) = W(i)$
  2.  If $\epsilon(i) \neq 0$, then $w_0(i+1) = \sum_{k \leq i}w_0(k)$ and $W(i+1) = W(i) + \eta\cdot \epsilon(i) \[1 + \frac{1}{2}\eta\cdot \epsilon(i) \] \cdot X^T_{i\bullet}$
That is, with Method 2, if the weight vector has to be updated, the difference between the two consecutive weight vectors is quadratic in the last prediction error, $\epsilon$, and the learning rate.
#### Method 3:
Here, the updating rules are:
  1. If $\epsilon(i) = 0$, then $w_0(i+1) = w_0(i) \, \text{and } \  W(i+1) = W(i)$
  2.  If $\epsilon(i) \neq 0$, then $w_0(i+1) = \sum_{k \leq i}w_0(k)$ and $W(i+1) = W(i) + \tanh\[\eta\cdot \epsilon(i) \cdot X^T_{i\bullet}\]$.

The hyperbolic tangent function here introduces a high degree of non-linearity in the updating of the weight vector. Note, however, that for $\eta \cdot \epsilon(i) X^T_{i\bullet} \to 0 $, Methods 2 and 3 essentially reduce to Method 1.

## Adaptive Linear Neuron (Adaline)
The Adaline approach was first introduced by Bernard Widrow and Tedd Hoff. Unlike the Rosenblatt's perceptron where the weight update is done sequentially sweeping through the samples in the dataset, here, the weights are updated as a batch in the form of a vector $\mathbf{W}$, reducing the process of weight updating to vector operations. Thus, whenever $\mathbf{W}$ is updated indicates a passing of an epoch. This is an important component of the reason why the Adaline method is much faster than Rosenblatt's perceptron. 

Furthermore, with Adaline the weights are updated using the following linear activation function (instead of a step-function, like in the perceptron model):
```math
\phi(\mathbf{Z}(k)) = \mathbf{Z}(k)
```
where, $\mathbf{Z}(k)$, the input value vector at the $k$-th epoch, is defined as:
```math
\mathbf{Z}(k) =  w_o(k) \cdot \mathbf{I}_{1 \times N} + \mathbf{W}(k) \cdot \mathbf{X}^T
```
where $w_o(k)$ is the bias unit at the $k$-th epoch, $\mathbf{W}(k)$ is the weight vector at the $k$-th epoch, $\mathbf{X}$ is the sample dataset and $\mathbf{I}_{1 \times N}$ is a $1 \times N$ vector with entries all equal to 1. Recall that $X$ is a $N \times M$ matrix, where each of the $M$ columns represents a feature (or degree of freedom) and each of the $N$ rows represents a sample. The weight vector $\mathbf{W}(k)$ is of dimension $1 \times M$, hence the input value vector $\mathbf{Z}(k)$ is of dimension $1 \times N$.

Like any supervised machine learning algorithm, we set aside a fraction of the target feature to use when training the algorithm. Let $\mathbf{Y}_{label}$ be the vector representing the target feature corresponding to the sample data $\mathbf{X}$ that we are going to use for the training. 

$\mathbf{Y}_{label}$ is, thus, a $1 \times N$ vector.

The role of the vector output of the activation function, $\phi(\mathbf{Z}(k))$, in the Adaline neural network is a very important one. Indeed, its entries are taken as equal to the neural network's prediction of the targeted feature for the corresponding sample. For example, at the $k$-th epoch, the $n$-th entry of  $\phi(\mathbf{Z}(k))$ is regarded as the neural network's prediction of the target feature for the $n$-th sample (or row) of $X$. If we indicate the neural network's prediction for the target feature at the $k$-th epoch as $\mathbf{Y}(k)_{predicted}$, then, for the Adaline neural network,
```math
\mathbf{Y}(k)_{predicted} = \phi(\mathbf{Z}(k)) \ .
```
With this in mind, the objective function employed is the sum of squared errors (SSE), $J\left(\mathbf{W}(k)\right)$:
```math
J\left(\mathbf{Y}(k) \right) = \frac{1}{2} \left( \mathbf{Y}_{label} - \mathbf{Y}(k)_{predicted} \right)^2 
```
which we may also rewrite as:
```math
J\left(\mathbf{W}(k) \right) = \frac{1}{2} \left( \mathbf{Y}_{label} - \phi(\mathbf{Z}(k)) \right)^2 = \frac{1}{2} \left( \mathbf{Y}_{label} - w_o(k) \cdot \mathbf{I}_{1 \times N} - \mathbf{W}(k) \cdot \mathbf{X}^T \right)^2
```
## Applications
We have applied our implementation to two classic classification problems: determining the severity ("severe" or "not severe") of the eruptions of the Old Faithful geyser (see file "example1.py") and the Iris classification problem (i.e., determining whether a flower is "Iris-setosa" or "Iris-versicolor").
