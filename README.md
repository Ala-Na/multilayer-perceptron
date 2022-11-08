<h1 align="center">Multilayer-Perceptron</h1>
<h2 align="center">Neural Network from Scratch</h2>


<p align='center'>
A 42 school project, from the machine learning / artificial intelligence branch.
 </p>

## Purpose
>A project to build a neural network from scratch.
>The goal is to train, then predict if a cell is malign and benign (using WDBC dataset, more infos inside ```data_infos.txt```).


## Program

There's only one program : ```python3 multilayer_perceptron.py``` with multiples options.
If not specified overwise, all options are compatible.

### 📊 --dataset
```python3 multilayer_perceptron.py --dataset path_to_data```
Option to specify the data path on which will be trained the model. Dataset path must be a valid ```.csv``` file with correct authorizations.
Note that it need to follow the same format (presence of id as index, labels either 'M' or 'B', same number of numerical features) as the default option ```data.csv``` to be functional.

### 🌾 --seed
```python3 multilayer_perceptron.py --seed number```
```number``` must be a valid integer. 
Randomness is used to initialize each neural layer. Using this option allow the user to get reproductible results (same weights if seed is kept).
By default, no seed is set.

### 📏 --metrics
```python3 multilayer_perceptron.py --metrics accuracy precision recall f1```
By default, only categorical cross entropy loss is displayed (both in terminal and in a final schema) and kept track of.
User can choose to display more metrics (on terminal and final schema) with this option. One or a combinaison of ```accuracy```, ```precision```, ```recall``` or ```f1``` (for F1 score) can be used.

### 🚅 --trainset_only
```python3 multilayer_perceptron.py --trainset_only```
By default, the dataset is cut in both training set (80% of dataset, to train model on) and validation set (20%, to see if model generalize well or not and not used for training) and metrics are display for both.
If this option is set, the cut isn't happening and model will be trained on all dataset.
Non compatible with ```--stop``` and ```--show_only``` options.

### ↩️ --epochs
```python3 multilayer_perceptron.py --epochs number```
```number``` must be a valid positive integer. 
Number of iterations/epochs on which the model is trained. An epoch comprise a forward propagation and a backward propagation (with weights update).
By default, 10000 epochs are performed.

### 📐 --dim
```python3 multilayer_perceptron.py --dim number_1 ... number_n```
```number_1 ... number_n``` must be valid positive integers, separated by spaces.
Option to set the hidden layers dimensions of the model. By default, an architecture with 2 hiddens layers of respectively 15 and 8 units is used.
Note that all those hidden layers will use a ReLU activation function.

## 📈 --opti
```python3 multilayer_perceptron.py --opti optimization_algorithm```
```optimization algorithm``` must either be ```rmsprop```, ```adam``` or ```momentum```. Each option correspond to the equivalent algorithm and will be applied to the full neural network.
By default, no optimization algorithm is used.

### 🔱 --alpha
```python3 multilayer_perceptron.py --alpha number```
```number``` must be a valid positive float included between 0 and 1. 
Choice of learning rate hyperparameter (value by which weights are proportionally updated). By default, alpha is set to 0.001.

### 📛 --name
```python3 multilayer_perceptron.py --name name```
```name``` must be a string.
Upon training, models are saved under pickle format. By default, the name is ```Model``` and a corresponding ```Model.pkl``` is created. This can be modified by this option.
Note that if used with ```---predict``` option, ```--name``` will correspond to the ```.pkl``` file from which the trained model will be retrieved.

### ✋ --stop
```python3 multilayer_perceptron.py --stop number```
```number``` must be a valid positive integer.
This option implement early stopping. Validation set loss in compared with it's previous best value (lower) after at least ```number``` epochs back. If an increase between previous and current value is seen, it may be because of a phenomenon called overfitting (model can't generalize well). Model go back to previous state where the smaller validation loss was met ad training stops.
Non compatible with ```--trainset_only``` option.

### 🏺 --lambda_
```python3 multilayer_perceptron.py --lambda_ number```
```number``` must be a valid positive float included between 0 and 1.
Regularization is a way to prevent overfitting (model not able to generalize to data it wasn't trained on).
If regularization is performed, it's factor (extent) must be set by this regularization factor. 
By default, L2 regularization is performed and ```lambda_``` is set to 0.5.



<p align='center'>
 <img width= '350' align='center' src='https://user-images.githubusercontent.com/67599180/194900103-66c28466-2930-44a4-94c8-d0f003784cdd.gif' alt='animated'>
</p>

<p align='center'>
 <i align='center'>Project screen capture of interactive mode</i>
</p>

### 🧮 describe.py
```python3 describe.py dataset_filepath```

A program which take as argument the filepath to a dataset to analyze and output corresponding statistical informations.
The goal is to reproduce the describe function of sklearn.
Statistical informations displayed: Count, mean, standard deviation, minimum, first quartile, median, third quartile, maximum, mode, range, interquartile range and number of outliers.

### 📊 histogram.py
```python3 histogram.py```
```python3 histogram.py -expl```

A program to display an histogram of marks distribution for each subject and each Hogwarts house.
The goal was to answer the question "Which subject at Hogwarts is homogeneously distributed between the four houses ?".
With ```-expl``` argument, explanations from the interactive scenario are displayed. 

<p align='center'>
 <img width= '700' align='center' src='https://user-images.githubusercontent.com/67599180/194900426-d29fa66b-db50-44ab-8f6b-80a02d745a35.png' alt='animated'>
</p>
<p align='center'>
 <i align='center'>Project screen shot</i>
</p>

### 📏 scatter_plot.py
```python3 scatter_plot.py```

A program to display a scatter plot of marks distribution for each pair of subjects, and each Hogwarts house.
The goal was to find the two most similars subjects.
With ```-expl``` argument, explanations from the interactive scenario are displayed. 

<p align='center'>
 <img width= '700' align='center' src='https://user-images.githubusercontent.com/67599180/194900436-d0afbc0f-bdd8-42e8-933c-7a5409bf481d.png' alt='animated'>
</p>
<p align='center'>
 <i align='center'>Project screen shot</i>
</p>

### 📐 pair_plot.py
```python3 scatter_plot.py```

A program to display a pair plot of marks distribution of all Hogwarts lessons subjects, for all Hogwarts house.
The goal was to choose useful features to train our model on.
With ```-expl``` argument, explanations from the interactive scenario are displayed. 

<p align='center'>
 <img width= '700' align='center' src='https://user-images.githubusercontent.com/67599180/194900432-d42f949e-4c10-4aec-882e-ffd65df64e54.png' alt='animated'>
</p>
<p align='center'>
 <i align='center'>Project screen shot</i>
</p>

### 🏋 logreg_train.py
```python3 logreg_train.py dataset_filepath```

A program to train our One-vs-All model on a correct Hogwarts dataset (```dataset_train.csv```) taken as argument.
The goal was to reach an accuracy of 98%.
Obtained weights are saved into ```thetas.npz```.
With ```-expl``` argument, explanations from the interactive scenario are displayed. 

### 🔮 logreg_predict.py
```python3 logreg_predict.py dataset_filepath```

A program to attribute Hogwarts' students in one of the four houses.
It takes a correct dataset as argument (```dataset_test.csv```), as well as a weights file (```thetas.npz```).
Predictions are saved into ```houses.csv```.
With ```-expl``` argument, explanations from the interactive scenario are displayed. 

### 📈 logreg_finetune.py
```python3 logreg_finetune.py```

A bonus program, to help find good hyperparameters to use as default value inside ```logreg_train.py```.
It trains the One-Vs-All model with randomly chosen hyperparameters and output the mean of ten differents training (with potential differents randomly initialized weights) with those hyperparameters.
100 experiments are run on the same training and testing sets. Results and hyperparameters values of those experiments are registered in ```experiments.csv```.
At the end of the program, median or mode of hyperparameters values helpful to reach best accuracy are displayed.

## Logistic regression details
A lot of available options make more sense for neural networks than linear regression. They can still help reaching faster optimal parameters, reduce overfitting, ... But their implementation was mainly to understand their logic.
### Available optimizations
- RMSprop (more neural network oriented)
- Momentum (more neural network oriented)
- Adam (more neural network oriented)
- Learning rate decay
- Mini-batch or stochastic gradient descent

### Available regularizations
- L2 regularization (ridge linear regression)
- Early stopping

## Available weights initializations
- Zeros
- Random small numbers (0-1)
- He initialization

### Available hyperparameters
- Max iter = Number of epochs
- Alpha = Learning rate
- Beta 1 = Value for momentum / Adam (more neural network oriented)
- Beta 2 = Value for RMSprop / Adam (more neural network oriented)
- Lambda = Value for L2 regularization
- Decay = Learning rate decay
- Decay interval = Interval to perform learning rate decay
- Epsilon = Small number to avoid computation problem in RMSprop / Adam
- Batch size = Size of a batch  (1 = stochastic < mini-batch < max size = batch)


## Language used
Python :snake:
<i>
Why ? Because it's the main language used in data science and machine learning nowadays.
 </i>


## Libraries used
- NumPy (version: 1.21.5)
- pandas (version 1.5.0)
- matplotlib (version 3.5.1)
- scikit-learn (version 1.1.2)
- playsound (version 1.3.0)
- argparse (version 1.1)

Note that seaborn could have been used for ```pair_plot.py``` but, as 42 restrict the quantity of memory per student, I choose to use solely matplotlib for visualization.
