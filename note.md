[Toc]

# Part 1 Supervised Learning

### Linear Regression

Regression model: Predicts numbers (Infinitely many possible outputs)

Classification model: Predicts categories (Small number of possible outputs)

<img src="images\Snipaste_2023-11-15_09-48-24.png" style="zoom:75%;" />

#### Cost Function

<img src="images\Snipaste_2023-11-15_09-39-09.png" style="zoom:75%;" />

#### Gradient Descent

<img src="images\Snipaste_2023-11-15_09-44-13.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-15_09-53-17.png" style="zoom:75%;" />

"Batch" Gradient descent: Each step of the gradient descent, uses all the training examples.

#### Learning rate

<img src="images\Snipaste_2023-11-15_09-50-48.png" style="zoom:75%;" />

#### Multiple Features

<img src="images\Snipaste_2023-11-15_09-56-28.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-15_09-56-55.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-15_09-57-20.png" style="zoom:75%;" />

#### Vectorization

<img src="images\Snipaste_2023-11-15_09-58-59.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-15_10-03-52.png" style="zoom:75%;" />

#### Gradient Descent for Multiple Regression

$$
w_j = w_j - \alpha \frac{\partial}{\partial w_j}J(\vec{w}, b) = w_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)})x^{(i)}_j
$$

$$
b = b - \alpha \frac{\partial}{\partial b}J(\vec{w}, b) 
$$



#### Practical Tips for Linear Regression

##### Feature Scaling

<img src="images\Snipaste_2023-11-15_10-18-20.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-15_10-18-43.png" style="zoom:75%;" />

![](images\Snipaste_2023-11-15_10-21-27.png)

##### Checking Gradient Descent for Convergence

<img src="images\Snipaste_2023-11-15_10-22-59.png" style="zoom:75%;" />

##### Choosing the learning rate

<img src="images\Snipaste_2023-11-15_10-25-33.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-15_10-26-00.png" style="zoom:75%;" />

##### Feature Engineering

<img src="images\Snipaste_2023-11-15_10-27-08.png" style="zoom:75%;" />

##### Polynomial Regression

<img src="images\Snipaste_2023-11-15_10-32-27.png" style="zoom:75%;" />



### Classification(Logistic Regression) 

#### Sigmoid function

![](images\Snipaste_2023-11-15_10-43-37.png)

#### Decision Boundary

<img src="images\Snipaste_2023-11-15_10-43-09.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_10-44-58.png" style="zoom:80%;" />

#### Cost function

![](images\Snipaste_2023-11-15_10-45-39.png)

#### Gradient descent

![](images\Snipaste_2023-11-15_11-04-24.png)

#### Regularization to Reduce Overfitting

##### The Problem of overfitting

![](images\Snipaste_2023-11-15_11-06-37.png)

![](images\Snipaste_2023-11-15_11-07-54.png)

##### Addressing Overfitting

- Collect more training examples
- Select features to include/exclude
- Regularization

![](images\Snipaste_2023-11-15_11-10-11.png)

![](images\Snipaste_2023-11-15_11-10-59.png)

![](images\Snipaste_2023-11-15_11-11-41.png)

#### Cost Function with Regularization

![](images\Snipaste_2023-11-15_11-13-38.png)

![](images\Snipaste_2023-11-15_11-14-31.png)

![](images\Snipaste_2023-11-15_11-15-49.png)

# Part 2 Advanced Learning Algorithms

## Week 1 - Week 2 Neural Networks

### Neural Network Introduction

#### Model

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-15_11-32-35.png">
</center>

#### Activation function

![](images\Snipaste_2023-11-15_11-39-20.png)

### Speculations on artificial general intelligence (AGI)

![](images\Snipaste_2023-11-15_11-46-11.png)

<img src="images\Snipaste_2023-11-15_11-47-16.png" style="zoom:75%;" />

### Vectorization

<img src="images\Snipaste_2023-11-15_11-48-51.png" style="zoom: 80%;" />

### Neural Network Training

![](images\Snipaste_2023-11-15_11-53-23.png)



<img src="images\Snipaste_2023-11-15_11-54-49.png"  />

<img src="images\Snipaste_2023-11-15_11-55-07.png"  />

![](images\Snipaste_2023-11-15_11-55-52.png)

### Multiclass Classification

![](images\Snipaste_2023-11-15_12-06-34.png)

![](images\Snipaste_2023-11-15_12-07-32.png)

![](images\Snipaste_2023-11-15_12-09-42.png)

![](images\Snipaste_2023-11-15_12-10-45.png)

![](images\Snipaste_2023-11-15_12-13-00.png)

![](images\Snipaste_2023-11-15_12-13-55.png)

### Multi-label Classification

![](images\Snipaste_2023-11-15_12-15-08.png)

![](images\Snipaste_2023-11-15_12-15-47.png)

### Advanced Optimization

![](images\Snipaste_2023-11-15_12-17-06.png)

![](images\Snipaste_2023-11-15_12-17-32.png)

## Week 3 Advice for applying machine learning

### Deciding what to try next

#### Machine learning diagnostic

![](images\Snipaste_2023-11-15_15-51-47.png)

High bias(Underfitting):

- Try getting additional features
- Try adding polynomial features ($x_1^2$, $x_2^2$, $x_1x_2$,*etc*)
- Try decreasing $\lambda$

High Variance(Overfitting):

- Get more training examples
- Try smaller sets of features
- Try increasing $\lambda$

### Evaluating and choosing models

<img src="images\Snipaste_2023-11-15_16-06-13.png" style="zoom:80%;" />

#### Inappropriate method:

![](images\Snipaste_2023-11-15_16-07-53.png)

#### recommended method

<img src="images\Snipaste_2023-11-15_16-08-50.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-09-10.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-09-42.png" style="zoom:80%;" />



### bias and variance

#### Diagnosing bias and variance

![](images\Snipaste_2023-11-15_16-11-33.png)

<img src="images\Snipaste_2023-11-15_16-11-59.png" style="zoom:80%;" />

#### Regularization

<img src="images\Snipaste_2023-11-15_16-14-20.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-15-10.png"  />

#### Establishing a baseline level of performance

What is the level of error you can reasonably hope to get to?

- Human level performance
- Competing algorithms performance
- Guess based on experience

<img src="images\Snipaste_2023-11-15_16-16-34.png" style="zoom:80%;" />

#### Learning curves

<img src="images\Snipaste_2023-11-15_16-19-56.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-23-10.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-23-38.png" style="zoom:80%;" />

#### Neural networks and bias variance

A large neural network will usually do as well or better than a smaller one so long as regularization is chosen appropriately.

![](images\Snipaste_2023-11-15_16-24-44.png)

![](images\Snipaste_2023-11-15_16-25-28.png)

### Machine learning development process

<img src="images\Snipaste_2023-11-15_16-26-12.png" style="zoom:80%;" />

#### Error analysis

![](images\Snipaste_2023-11-15_16-28-37.png)

#### Adding data

<img src="images\Snipaste_2023-11-15_16-34-11.png" style="zoom:80%;" />

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-15_16-34-39.png" width=440 />
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-15_16-35-19.png" width=440 />
</center>

<img src="images\Snipaste_2023-11-15_16-35-58.png" style="zoom:80%;" />



![](images\Snipaste_2023-11-15_16-37-20.png)

![](images\Snipaste_2023-11-15_16-37-37.png)

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-15_16-38-08.png" width="450">
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-15_16-39-06.png" width="200">
</center>

#### Transfer learning: using data from a different task

<img src="images\Snipaste_2023-11-15_16-45-31.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-48-11.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-48-32.png" style="zoom:80%;" />

#### Full cycle of a machine learning project

![](images\Snipaste_2023-11-15_16-51-02.png)

### Skewed datasets

<img src="images\Snipaste_2023-11-15_16-52-10.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-52-31.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-53-19.png" style="zoom:80%;" />

<img src="images\Snipaste_2023-11-15_16-53-45.png" style="zoom:80%;" />



## Week 4 Decision Trees

### Decision Trees

#### Decision Tree Learning

**Decision 1:** How to choose what feature to split on at each node?

- Maximize purity (or minimize impurity)

![](images\Snipaste_2023-11-13_10-35-30.png)

**Decision 2:** When do you stop splitting?

- When a node is 100% one class
- When splitting a node will result in the tree exceeding a maximum depth
- When improvements in purity score are below a threshold
- When number of examples in a node is below a threshold



#### Measuring purity

- **Entropy(熵)** as a measure of impurity

![](images\Snipaste_2023-11-13_10-38-12.png)

![](images\Snipaste_2023-11-13_10-38-49.png)



#### Choosing a split: Information Gain

![](images\Snipaste_2023-11-13_10-40-57.png)



#### Putting it together

1. Start with all examples at the root node 
2. Calculate information gain for all possible features, and pick the one with the highest information gain
3. Split dataset according to selected feature, and create left and right branches of the tree
4. Keep repeating splitting process until stopping criteria is met:
   - When a node is 100% one class
   - When splitting a node will result in the tree exceeding a maximum depth
   - Information gain from additional splits is less than threshold
   - When number of examples in a node is below a threshold



#### Using one-hot encoding of categorical features

<center class='half'>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_10-49-18.png" width="470">
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_10-49-58.png" width="470">
</figure>

If a categorical feature can take on $k$ values, create $k$ binary features (0 or 1 valued).

<img src="images\Snipaste_2023-11-13_10-53-34.png" style="zoom:50%;" />



#### Continuous valued features

<img src="images\Snipaste_2023-11-13_10-58-13.png" style="zoom: 50%;" />

<img src="images\Snipaste_2023-11-13_10-59-31.png" style="zoom:50%;" />



#### Regression Trees

<img src="images\Snipaste_2023-11-13_11-05-08.png" style="zoom:50%;" /> **Input X, Output y**

<img src="images\Snipaste_2023-11-13_11-05-57.png" style="zoom:50%;" />每个叶子节点的值为分给该叶子节点的所有样本(体重)的**平均值**

<img src="images\Snipaste_2023-11-13_11-06-43.png" width="650"> **Variance** as a measure of impurity



### Tree Ensembles

#### Using multiple decision trees

Reason: Trees are highly sensitive to small changes of the data

<img src="images\Snipaste_2023-11-13_11-16-16.png" style="zoom:50%;" />



#### Sampling with replacement

<img src="images\Snipaste_2023-11-13_11-17-50.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_11-18-12.png" style="zoom:50%;" />



#### Random forest algorithm

<img src="images\Snipaste_2023-11-13_11-20-00.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_11-20-38.png" style="zoom:50%;" />



#### XGBoost

<img src="images\Snipaste_2023-11-13_11-21-41.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_11-24-08.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_11-24-46.png" style="zoom:50%;" />



### Conclusion

<img src="images\Snipaste_2023-11-13_11-28-35.png" style="zoom:50%;" />



# Part 3 Unsupervised Learning, Recommenders, Reinforcement Learning

## Week1 Unsupervised learning

### Clustering

#### What is clustering

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_11-36-00.png" width="450">
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_11-36-12.png" width="450">
</center>



#### K-means algorithm

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_11-40-15.png" width="450">
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_11-40-25.png" width="450">
</center>



<img src="images\Snipaste_2023-11-13_11-46-41.png" style="zoom:50%;" />



#### Optimization objective

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_11-51-49.png" width="476" />
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_11-59-34.png" width="476" />
</center>



#### Initializing K-means

<img src="images\Snipaste_2023-11-13_12-01-48.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-02-46.png" alt="Snipaste_2023-11-13_12-02-46" style="zoom:40%;" /> Different initializations result in different clusters 

<img src="images\Snipaste_2023-11-13_12-03-36.png" style="zoom:50%;" />



#### Choosing the Number of Clusters

<img src="images\Snipaste_2023-11-13_12-07-28.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-08-05.png" style="zoom:50%;" />



### Anomaly Detection

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_12-10-00.png" width="470">
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_12-10-38.png" width="470">
</center>

<img src="images\Snipaste_2023-11-13_12-12-13.png" style="zoom:50%;" />



#### Gaussian (Normal) Distribution

<img src="images\Snipaste_2023-11-13_12-12-53.png" alt="Snipaste_2023-11-13_12-12-53" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-14-41.png" style="zoom:50%;" />



#### Algorithm

<img src="images\Snipaste_2023-11-13_12-17-50.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-18-45.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-20-28.png" style="zoom:50%;" />



#### Developing and evaluating an anomaly detection system

<img src="images\Snipaste_2023-11-13_12-21-41.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-27-11.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-29-26.png" style="zoom:50%;" />



#### Anomaly detection vs. supervised learning

<img src="images\Snipaste_2023-11-13_12-36-49.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-37-18.png" style="zoom:50%;" />



#### Choosing what features to use

<img src="images\Snipaste_2023-11-13_12-38-23.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-40-16.png" style="zoom:50%;" />

<img src="images\Snipaste_2023-11-13_12-40-23.png" style="zoom:50%;" />



## Week2 Recommender System

### Collaborative Filtering

#### Problem description

<img src="images\Snipaste_2023-11-13_16-50-33.png" style="zoom:70%;" />

<img src="images\Snipaste_2023-11-13_17-05-26.png" style="zoom:75%;" />



#### Cost function

<img src="images\Snipaste_2023-11-13_17-06-29.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-13_17-07-43.png" style="zoom:75%;" />



#### From regression to binary classification

<img src="images\Snipaste_2023-11-13_17-10-53.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-13_17-11-40.png" style="zoom:75%;" />



#### Mean normalization

Suppose there is a new user who has never rated any movies, and we will train $w$,$b$, and $x$ using the loss function below. Since the user has never rated any movies, the user's parameters $w$ and $b$ will not appear in the first term of the loss function, which leads to the optimization of the loss function to make the user's parameter $w$ as small as possible.Therefore, the user's parameter $w$ will be set to 0, resulting in all rating predictions for this user being 0.This is what we don't want to see.

<img src="images\Snipaste_2023-11-13_17-29-11.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-13_17-33-14.png" style="zoom:75%;" />



#### TensorFlow implementation

<img src="images\Snipaste_2023-11-13_17-36-21.png" style="zoom:75%;" />



#### Finding related items

<img src="images\Snipaste_2023-11-13_17-42-26.png" style="zoom:75%;" />

<img src="images\Snipaste_2023-11-13_17-42-52.png" style="zoom:75%;" />



### Content-based Filtering

<img src="images\Snipaste_2023-11-13_18-56-45.png" style="zoom:67%;" />

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_19-17-44.png" width="460">
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_19-19-58.png" width="460">
</center>



#### Deep learning for content-based filtering

> <center>
>     <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_19-22-28.png" width="460"/>
>     <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_19-23-16.png" width="460"/>
> </center>
>
> <img src="images\Snipaste_2023-11-13_19-25-49.png" style="zoom:75%;" />



#### Recommending from a large catalogue

How to efficiently find recommendation from a large set of items?

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_19-39-39.png" width="460" />
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-13_19-41-27.png" width="460" />
</center>



- Retrieving more items results in better performance, but slower recommendations.
- To analyse/optimize the trade-off, carry out offline experiments to see if retrieving additional items results in more relevant recommendations (i.e., $p(y^{(i,j)} = 1)$  of items displayed to user are higher).



#### TensorFlow Implementation

<img src="images\Snipaste_2023-11-13_19-49-14.png" style="zoom:75%;" />



## Week3 Reinforcement Learning

#### What is Reinforcement Learning?

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-14_10-33-31.png" width="460">
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-14_10-38-34.png" width="460">
</center>

**Reinforcement Learning**  vs  **Supervised Learning**

- Less historical data is required for reinforcement learning, instead, we need to try different actions to construct the training data
- Reinforcement Learning doesn't need any labeled data, which means the learner doesn't tell us what the correct action should be for each step
- In the process of reinforcement learning, the reward signal is delayed, that is, the environment will tell us much later whether the action taken before is effective or not



#### Key concepts in reinforcement learning

<img src="images\Snipaste_2023-11-14_10-54-46.png" style="zoom:75%;" />

- A policy is a function $\pi(s) = a$ mapping from states to actions, that tells you what action *a* to take in a given state *s*.
- The goal of reinforcement learning: Find a policy $\pi$ that tells you what action $(a = \pi(s))$ to take in every state (s) so as to maximize the return.

<img src="images\Snipaste_2023-11-14_10-58-34.png" style="zoom:75%;" />



#### State-action value function

<img src="images\Snipaste_2023-11-14_10-59-52.png" style="zoom:75%;" />

The best possible return from state $s$ is $\underset{a}{max}Q(s, a)$.

The best possible action in state $s$ is the action $a$ that gives $\underset{a}{max}Q(s, a)$.



#### Bellman Equation

$s$ : current state

$a$ : current action

$s^{'}$ : state you get to after taking action $a$

$a^{'}$ : action that you take in state $s^{'}$
$$
Q(s, a) = R(s) + \gamma * \underset {a^{'}}{max}Q(s^{'},a^{'})
$$
$R(s)$ : Reward you get right away

$\underset {a^{'}}{max}Q(s^{'},a^{'})$ : Return from behaving optimally starting from state $s^{'}$



#### Random (stochastic) environment

<img src="images\Snipaste_2023-11-14_11-24-26.png" style="zoom:75%;" />

Expected Return = $Average(R1 + \gamma R2 + \gamma^2 R3 + ···)$ = $E(R1 + \gamma R2 + \gamma^2 R3 + ···)$

$Q(s, a) = R(s) + \gamma * E[\underset {a^{'}}{max}Q(s^{'},a^{'})]$  



#### Continuous State Spaces

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-14_11-39-54.png" width="460" />
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-14_11-41-48.png" width="460" />
</center>



#### Learning the state-value function

<img src="images\Snipaste_2023-11-14_11-43-37.png" style="zoom:75%;" />

#### Improved neural network architecture

<img src="images\Snipaste_2023-11-14_11-50-04.png" style="zoom:75%;" />

#### Learning algorithm

<img src="images\Snipaste_2023-11-14_11-53-50.png" style="zoom:75%;" />

#### Algorithm refinement: $\varepsilon$ - greedy policy

How to choose actions while still learning?

Option 1：

​	Pick the action $a$ that maximizes $Q(s, a)$.          ---------------------------------will never try other actions

Option 2 :

​	With probability 0.95, pick the action $a$ that maximizes $Q(s, a)$.   -------greedy exploitation

​	With probability 0.05, pick an action $a$ randomly.  ----------------------------exploration

​	$\varepsilon$ - greedy policy ($\varepsilon = 0.05$)

​		greedy 95% of the time

​		exploring 5% of the time

​		start $\varepsilon$ high, gradually decrease

#### Target Network

why do we need target network?

- using neural networks in reinforcement learning to estimate action-value functions has proven to be highly unstable.

We can train the $Q$-Network by adjusting it's weights at each iteration to minimize the mean-squared error in the Bellman equation, where the target values are given by:

$$
y = R + \gamma \max_{a'}Q(s',a';w)
$$

where $w$ are the weights of the $Q$-Network. This means that we are adjusting the weights $w$ at each iteration to minimize the following error:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}Q(s',a'; w)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

Notice that this forms a problem because the $y$ target is changing on every iteration. Having a constantly moving target can lead to oscillations and instabilities. To avoid this, we can create
a separate neural network for generating the $y$ targets. We call this separate neural network the **target $\hat Q$-Network** and it will have the same architecture as the original $Q$-Network. By using the target $\hat Q$-Network, the above error becomes:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}\hat{Q}(s',a'; w^-)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

where $w^-$ and $w$ are the weights of the target $\hat Q$-Network and $Q$-Network, respectively.

In practice, we will use the following algorithm: every $C$ time steps we will use the $\hat Q$-Network to generate the $y$ targets and update the weights of the target $\hat Q$-Network using the weights of the $Q$-Network. We will update the weights $w^-$ of the the target $\hat Q$-Network using a **soft update**. This means that we will update the weights $w^-$ using the following rule:

$$
w^-\leftarrow \tau w + (1 - \tau) w^-
$$

where $\tau\ll 1$. By using the soft update, we are ensuring that the target values, $y$, change slowly, which greatly improves the stability of our learning algorithm.



#### Algorithm refinement: Mini-batch and soft update

Mini-batch

<center>
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-14_12-04-27.png" width="460" />
    <img src="E:\课程\Machine_learning_spelization\images\Snipaste_2023-11-14_12-05-07.png" width="460" />
</center>

![](images\Snipaste_2023-11-14_12-07-34.png)

soft update (used for target Q network)

<img src="images\Snipaste_2023-11-14_12-08-16.png" style="zoom:75%;" />



#### Experience Replay

When an agent interacts with the environment, the states, actions, and rewards the agent experiences are **sequential** by nature. If the agent tries to learn from these consecutive experiences it can run into problems due to the strong correlations between them. To avoid this, we employ a technique known as **Experience Replay** to generate uncorrelated experiences for training our agent. Experience replay consists of storing the agent's experiences (i.e the states, actions, and rewards the agent receives) in a memory buffer and then sampling a random mini-batch of experiences from the buffer to do the learning. The experience tuples $(S_t, A_t, R_t, S_{t+1})$ will be added to the memory buffer at each time step as the agent interacts with the environment.

By using experience replay we avoid problematic correlations, oscillations and instabilities. In addition, experience replay also allows the agent to potentially use the same experience in multiple weight updates, which increases data efficiency.



#### Final algorithm

<img src="images\deep_q_algorithm.png" style="zoom:40%;" />

