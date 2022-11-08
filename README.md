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

#### 📊 --dataset
```python3 multilayer_perceptron.py --dataset path_to_data```
>Option to specify the data path on which will be trained the model. Dataset path must be a valid ```.csv``` file with correct authorizations.
>Note that it need to follow the same format (presence of id as index, labels either 'M' or 'B', same number of numerical features) as the default option ```data.csv``` to be functional.

#### 🌾 --seed
```python3 multilayer_perceptron.py --seed number```
>```number``` must be a valid integer. 
>Randomness is used to initialize each neural layer. Using this option allow the user to get reproductible results (same weights if seed is kept).
>By default, no seed is set.

#### 📏 --metrics
```python3 multilayer_perceptron.py --metrics accuracy precision recall f1```
>By default, only categorical cross entropy loss is displayed (both in terminal and in a final schema) and kept track of.
>User can choose to display more metrics (on terminal and final schema) with this option. One or a combinaison of ```accuracy```, ```precision```, ```recall``` or ```f1``` (for F1 score) can be used.

#### 🚅 --trainset_only
```python3 multilayer_perceptron.py --trainset_only```
>By default, the dataset is cut in both training set (80% of dataset, to train model on) and validation set (20%, to see if model generalize well or not and not used for training) and metrics are display for both.
>If this option is set, the cut isn't happening and model will be trained on all dataset.
>Non compatible with ```--stop``` and ```--show_only``` options.

#### ↩️ --epochs
```python3 multilayer_perceptron.py --epochs number```
>```number``` must be a valid positive integer. 
>Number of iterations/epochs on which the model is trained. An epoch comprise a forward propagation and a backward propagation (with weights update).
>By default, 10000 epochs are performed.

#### 📐 --dim
```python3 multilayer_perceptron.py --dim number_1 ... number_n```
>```number_1 ... number_n``` must be valid positive integers, separated by spaces.
>Option to set the hidden layers dimensions of the model. By default, an architecture with 2 hiddens layers of respectively 15 and 8 units is used.
>Note that all those hidden layers will use a ReLU activation function.

#### 📈 --opti
```python3 multilayer_perceptron.py --opti optimization_algorithm```
>```optimization algorithm``` must either be ```rmsprop```, ```adam``` or ```momentum```. Each option correspond to the equivalent algorithm and will be applied to the full neural network.
>By default, no optimization algorithm is used.

#### 🔱 --alpha
```python3 multilayer_perceptron.py --alpha number```
>```number``` must be a valid positive float included between 0 and 1. 
>Choice of learning rate hyperparameter (value by which weights are proportionally updated). By default, alpha is set to 0.001.

#### 📛 --name
```python3 multilayer_perceptron.py --name name```
>```name``` must be a string.
>Upon training, models are saved under pickle format. By default, the name is ```Model``` and a corresponding ```Model.pkl``` is created. This can be modified by this option.
>Note that if used with ```---predict``` option, ```--name``` will correspond to the ```.pkl``` file from which the trained model will be retrieved.

#### ✋ --stop
```python3 multilayer_perceptron.py --stop number```
>```number``` must be a valid positive integer.
>This option implement early stopping. Validation set loss in compared with it's previous best value (lower) after at least ```number``` epochs back. If an increase between previous and current value is seen, it may be because of a phenomenon called overfitting (model can't generalize well). Model go back to previous state where the smaller validation loss was met ad training stops.
>Non compatible with ```--trainset_only``` option.

#### 🏺 --lambda_
```python3 multilayer_perceptron.py --lambda_ number```
>```number``` must be a valid positive float included between 0 and 1.
>Regularization is a way to prevent overfitting (model not able to generalize to data it wasn't trained on).
>If regularization is performed, it's factor (extent) must be set by this regularization factor. 
>By default, L2 regularization is performed and ```lambda_``` is set to 0.5.


#### 🗠 --show_all
```python3 multilayer_perceptron.py --show_all```
>By default, after training, a final schema of displayed metrics evolution is shown.
>If this option is set, training is not happening and all metrics evolution of previously trained models (saved in ```experiments.csv```) are displayed on a schema.
>Won't display anything if ```experiments.csv``` is not present, have incorrect rights or is not under expected format.
>This option may present some python exceptions.
>Non compatible with ```--show_only``` option.

#### 📉 --show_only
```python3 multilayer_perceptron.py --show_only```
>By default, after training, a final schema of displayed metrics evolution is shown.
>If this option is set, an additionnal schema of all metrics evolution of previously trained models (saved in ```experiments.csv```) is displayed.
>Won't display anything if ```experiments.csv``` is not present, have incorrect rights or is not under expected format.
>This option may present some python exceptions.
>Non compatible with ```--show_all``` option.

#### 🔮 --predict
```python3 multilayer_perceptron.py --predict path_to_data```
>By default, program perform training. If this option is set, it'll perform prediction instead on the dataset indicated by ```path_to_data```. Dataset path must be a valid ```.csv``` file with correct authorizations.
>Note that it need to follow the same format (presence of id as index, labels either 'M' or 'B', same number of numerical features) as the default option ```data.csv``` to be functional.
>Predicted labels are displayed, as well as the loss and others metrics (accuracy, precision, recall, F1 score).
>Non compatible with all options except ```--name--```option.

#### 🧹 --reset
```python3 multilayer_perceptron.py --reset```
>If set, delete ```experiments.csv``` file and all saved models in ```.pkl``` files at program launch.

#### 🚨 --reg
```python3 multilayer_perceptron.py --reg regularization```
>```regularization``` must be either ```l2``` or ```None```.
By default, set to ```l2```, which will make model perform regularization (a way to limit a phenomenon called overfitting, where model can't generalize well on data it wasn't trained on). Must be used with a ```--lambda_``` option different of 0.0.
If set to ```None```, no regularization will be performed.

## Additional program
```evaluation.py``` is a program generating randomly a ```data_traning.csv``` and ```data_test.csv``` from a remote data set hosted on 42 website. This set is the same as ```data.csv```.
During evaluation, this script was launched multiple times and loss was compared to a point scale.
Note that model performance can be altered by the randomness of the testing set generated. That's why multiple prediction on differents testing set are performed.

## Words on neural network implementation
I choose to make a "modular" implementation of neural network.
A ```SimpleNeuralNetwork``` class is present, which is a multilayer perceptron implementation containing a multiple number of ```DenseLayer``` class elements.
```DenseLayer``` is defined by multiple parameters (feel free to look at its implementation), including its activation function. By default, the activation function of all hidden layers was ReLU and the output layer activation was softmax in ```multilayer_perceptron.py``` program. But those parameters can be modified at ```DenseLayer``` intialization. Algorithm for forward and backward propagation, as well as update, are both presents in ```DenseLayer``` object (for one layer) and ```SimpleNeuralNetwork``` object (for all layers, calling methods of layers).
Though only a set of those objects parameters are modified in the main program, a lot of others options are available. Look it up !

<p align='center'>
 <img width= '350' align='center' src='https://user-images.githubusercontent.com/67599180/194900103-66c28466-2930-44a4-94c8-d0f003784cdd.gif' alt='animated'>
</p>

<p align='center'>
 <i align='center'>Project screen capture of interactive mode</i>
</p>


<p align='center'>
 <img width= '700' align='center' src='https://user-images.githubusercontent.com/67599180/194900426-d29fa66b-db50-44ab-8f6b-80a02d745a35.png' alt='animated'>
</p>
<p align='center'>
 <i align='center'>Project screen shot</i>
</p>





## Language used
Python :snake:
<i>
Why ? Because it's the main language used in data science and machine learning nowadays.
 </i>


## Libraries used
- NumPy (version: 1.21.5)
- pandas (version 1.5.0)
- matplotlib (version 3.5.1)
- argparse (version 1.1)
- pickle (version 4.0)

Note that metrics results where compared with equivalent TensorFlow functions.
