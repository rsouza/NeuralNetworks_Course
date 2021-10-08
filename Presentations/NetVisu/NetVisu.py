#import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd

#creates different dataset
import datasets

def persistence_homology(data, pers_dis = 1, intrinsic_dimension = 2):
    '''
    Plots the persistence barcodes for any dataset data, calculated until max edge distance pers_dis and up to dimension intrinsic_dimension
    Inputs: data: dataset as a point cloud; pers_dis = 1: maximum edge distance; intrinsic_dimension = 2: maximum homology dimension
    '''
    rips = gudhi.RipsComplex(points = data, max_edge_length = pers_dis)
    st = rips.create_simplex_tree(max_dimension = intrinsic_dimension)

    barcodes = st.persistence(homology_coeff_field = 2)
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1); ax2 = fig.add_subplot(1,2,2)
    gudhi.plot_persistence_barcode(barcodes, axes = ax1)
    gudhi.plot_persistence_diagram(barcodes, axes = ax2)
    plt.show()
    return 

#defines network based functions
def feedforward(number_of_layers, neurons_per_hidden_layer, activation, inputs_shape = 3):
    '''
    This function creates a simple feedforward neural network
    Input: number_of_layers: number of hidden layers in the network; neurons_per_hidden_layer: width of each hidden layer, which must be the
    same for all layers; activation: activation function for each hidden layer, which must be the same; inputs_shape = 3: shape of input
    set to 3 by default
    Output model of the neural network
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(inputs_shape,)))
    for i in range(number_of_layers):
        model.add(tf.keras.layers.Dense(neurons_per_hidden_layer, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def trains_network(model, number_of_epochs, data, target, verbose = 2, plot = False):
    '''
    Splits the dataset into train and validation and trains neurak network
    Input: model: the neural network to be trained; number_of_epochs: number of training epochs; data: dataset; target: dataset's labels;
    test_size = 0.15: selects percentage of dataset to be used as validation data, set to 0.15; verbose = 2: verbose in training history;
    plot == False: if True, plots the training history
    '''
    X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.15, random_state=42)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs = number_of_epochs, validation_data =(X_val, y_val), verbose = verbose)
    model.evaluate(data, target)
    
    if plot == True:
        pd.DataFrame(history.history).plot(figsize =(8, 5))
        plt.grid(True) 
        plt.gca().set_ylim(0, 1) 
        plt.show()
    return 

#defines the function that plots the PCAed output of each layer
def pca_network(model, data, target):
    '''
    Plots the pca output of each hidden layer to 3 dimensions (if the dimensions are bigger than 3) for a neural network model
    Input: nmodel: Feedforward Neural Network model; data: dataset; target: dataset's labels
    '''
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights()) 
    model_copy.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model_copy.evaluate(data, target)
    
    pca = PCA(n_components=3)

    for i in range(len(model_copy.layers)-1):
        layer = tf.keras.models.Model(inputs=model_copy.input,outputs=model_copy.layers[i].output)
        layer_output = layer.predict(data)
        pca_data= pca.fit_transform(layer_output)
        fig = plt.figure()
        ax = Axes3D(fig)

        red = pca_data[target == 0]
        blue = pca_data[target == 1]
        scatter = ax.scatter(red[:,0], red[:,1], red[:,2],c='r')
        scatter = ax.scatter(blue[:,0], blue[:,1], blue[:,2],c='b')
        ax.text2D(0.05, 0.95, "Hidden Layer "+str(i+1), transform=ax.transAxes)
        plt.show()

#define function that plots persistence homology study of each layer output

def persis_homol_network(model, data, target, intrinsic_dimension = 2, pers_dis = 1):
    '''
    Plots the persitence homology barcodes and diagrams of each layer of the neural network
    Input:  model: Feedforward Neural Network model; data: dataset; target: dataset's 
    labels; intrinsic_dimension = 1: intrinsic manifold dimension, so we know up to where perform the persistence homology;
    per_dis = 1: maximum distance to calculate persistence homology, 
    '''
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights()) 
    model_copy.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model_copy.evaluate(data, target)
    
    
    for i in range(len(model_copy.layers)-1):
        layer = tf.keras.models.Model(inputs=model_copy.input,outputs=model_copy.layers[i].output)
        layer_output = layer.predict(data)
        rips = gudhi.RipsComplex(points = layer_output, max_edge_length = pers_dis)
        st = rips.create_simplex_tree(max_dimension = intrinsic_dimension)

        barcodes = st.persistence(homology_coeff_field = 2)
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(1,2,1); ax2 = fig.add_subplot(1,2,2)
        gudhi.plot_persistence_barcode(barcodes, axes = ax1)
        gudhi.plot_persistence_diagram(barcodes, axes = ax2)
        fig.suptitle('Hidden Layer'+str(i+1))
        plt.show()
    
