# --- Dependencies

# Tensorfow 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input

# NumPy
import numpy as np

# Use the Cleverhans implementation of MI-FGSM
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method

# --- MI-FGSM Attack

# Implementation of the momentum iterative fast gradient sign method (MI-FGSM)
# Paper: Boosting Adversarial Attacks with Momentum
# Authors: Dong et al., 2017
class MomentumAttack:
    
    def __init__(self):
        pass
    
    # Perform the attack
    def attack(self, dataset, model, eps, eps_iter, nb_iter, norm, decay_factor):
        # Split labels from images
        y = dataset.unbatch().map(lambda x, y: y)
    
        # create an adversarial dataset with MI-FGSM
        x_adv = np.empty(shape=(0,64,64,3))
        for x_batch, y_batch in dataset:
            x_adv = np.concatenate((x_adv, momentum_iterative_method(model, x_batch, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm, decay_factor=decay_factor, clip_min=0, clip_max=255))) 
    
        # return pertubed dataset 
        return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x_adv),  y))

    # generate a dataset from the attack and evalute for used model (sanity checks)
    def generateDataset(self, folder,  xy, model_substitute, eps=8, eps_iter=0.8, nb_iter=10, norm=np.inf, decay_factor=1, spec = None):
        # the folder to store the dataset
        # choose a subdirectory for different epsilon values
        path = "datasets/adversarial_datasets/"+folder
        if spec != None:
            path += "/"+spec
            
        # Sanity check for the subsitute Model. Should be 100% when called with clean_dataset and substitute.
        print("After removing wrong classified examples:")
        model_substitute.evaluate(xy)
        
        # After adversarial pertubation
        x_adversarial = self.attack(xy, model_substitute, eps, eps_iter, nb_iter, norm, decay_factor)
        print("On adversarial dataset:")
        model_substitute.evaluate(x_adversarial.batch(128))
        
        # Save dataset
        print("Saving dataset")
        x_adversarial = x_adversarial.batch(128)
        tf.data.experimental.save(x_adversarial, path)
        print("Saved")
