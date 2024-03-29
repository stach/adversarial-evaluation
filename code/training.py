# --- Dependencies

# Tensorfow 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.applications import *
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

# NumPy and other
import numpy as np
import time

"""
    Generate an EfficientNetV2B3 base model 
"""    

class BaseModelGenerator:

    def __init__(self):
        pass

    def generateBaseModel(self, training_dataset, validation_dataset, name="test", epochs=100):
                
        # -------------------
        # --- Build model ---
        # -------------------
    
        # Load EfficientNetV2B3 with 128X128 image input (upscaled)
        input_shape = (64, 64, 3)
        inputs = Input(shape=input_shape)
        upscale = (keras.layers.Lambda(lambda x: tf.image.resize(x, [128, 128])))(inputs)
        model_base = EfficientNetV2B3(include_top=False, input_shape=input_shape, pooling=max, input_tensor=upscale) 
    
        # Create training model and append Dense layers
        model = tf.keras.models.Sequential()
        model.add(model_base)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(200, activation="softmax"))

        model.build()
        
        # ----------------------
        # --- Build training ---
        # ----------------------

        # For simpler loading in evaluation
        model.compile(metrics=["accuracy"])
        
        loss = tf.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        
        path = "trained_models/"+name+".h5"
        
        # Define our loss and accuracy metrics
        train_loss = tf.metrics.Mean(name="train_loss")
        train_accuracy = tf.metrics.CategoricalAccuracy(name="train_accuracy")
        
        val_loss = tf.metrics.Mean(name="val_loss")
        val_accuracy = tf.metrics.CategoricalAccuracy(name="val_accuracy")

        # Performs a training step on a batch
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                
                predictions = model(x)
                loss_emp = loss(y, predictions)
                
            gradients = tape.gradient(loss_emp, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update metrics for training set
            train_loss.update_state(loss_emp)
            train_accuracy.update_state(y, predictions)
            
        
        # -------------------
        # --- Train model ---
        # -------------------
        
        trained_epochs = 0
        epochs_time = []
        path = "trained_models/"+name+".h5" # where to save the model
        
        val_loss_prev = np.inf # track if val loss improved
        patience = 2 # for early stopping
            
        for epoch in range(epochs):
            print("Epoch ", (epoch+1) , "/", epochs)
            epoch_start = time.time()
            progress_bar_train = tf.keras.utils.Progbar(80000)
            
            # training
            for (x, y) in training_dataset:
                train_step(x, y)
                progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result()),("accuracy", train_accuracy.result())])
                
            epoch_end = time.time()
            trained_epochs+=1
            epochs_time.append(round(epoch_end-epoch_start))
                
            # calculate validation accuracy and loss after each epoch
            for (x, y) in validation_dataset:
                predictions = model(x)
                loss_emp = loss(y, predictions)
                val_loss.update_state(loss_emp)
                val_accuracy.update_state(y, predictions)
            
            print("Validation accuracy ", val_accuracy.result().numpy(), " and loss ", val_loss.result().numpy())
            
            # Feature #1 - Save model, if validation loss improved (threshold 0.25)
            # Feature #2 - Early stopping (2 iterations without val loss improves)
            
            if(val_loss.result().numpy()<val_loss_prev+0.025):
                model.save(path)
                print("Validation loss improved.")
                patience = 2
                val_loss_prev = val_loss.result().numpy()
            else:
                patience -= 1 
                print("Validation loss did not improved, patience: ", patience , ".")
                if(patience==0):
                    print("Training stopped!")
                    break
                    
        # Save the time from start to end of training            
        file = open("time.txt", "a")
        file.write("###\n")
        file.write(f"Model: {name}\n")
        file.write(f"Epochs: {trained_epochs}\n")
        file.write(f"Epochs time: {epochs_time}\n")
        file.write(f"Mean epoch time: {np.mean(epochs_time)}\n")
        file.close()

