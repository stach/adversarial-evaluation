# --- Dependencies

# Tensorfow 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input

# NumPy
import numpy as np

# --- Random Noise Attack

# Implementation of a l_inf bounded random noise attack
class NoiseAttack:
    
    def __init__(self):
        pass
    
    # Perform the attack
    def attack(self, dataset, eps):
        # Split labels from images
        y = dataset.unbatch().map(lambda x, y: y)
        
        # create an adversarial dataset with the direction of random gaussian noise
        x_adv = np.empty(shape=(0,64,64,3))
        for x_batch, y_batch in dataset:
            x_noise = x_batch+eps*np.sign(np.random.normal(0,1,(x_batch.shape[0],64,64,3)))
            x_adv = np.clip(x_adv, 0, 255)
            x_adv = np.concatenate((x_adv, x_noise)) 
    
        # return pertubed dataset 
        return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x_adv),  y))

    # generate a dataset from the attack and evalute for used model (sanity checks)
    def generateDataset(self, xy, model_substitute, eps):
        # the folder to store the dataset
        path = "datasets/evaluation_noise/"

        # Sanity check for the subsitute Model. Should be 100% when called with clean_dataset and substitute.
        print("After removing wrong classified examples:")
        model_substitute.evaluate(xy)
        
        # After adversarial pertubation
        x_adversarial = self.attack(xy, eps)
        print("On adversarial dataset:")
        model_substitute.evaluate(x_adversarial.batch(128))
        
        # Save dataset
        print("Saving dataset")
        x_adversarial = x_adversarial.batch(128)
        tf.data.experimental.save(x_adversarial, path)
        print("Saved")
