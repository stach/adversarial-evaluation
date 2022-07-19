# --- Dependencies

# Tensorfow 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input

# NumPy
import numpy as np

# Use the Cleverhans implementation of FGSM
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

# --- FGSM Attack

# Implementation of the fast gradient sign method (FGSM)
# Paper: Explaining and Harnessing Adversarial Examples
# Authors: Goodfellow et al., 2014
class FGSMAttack:
    
    def __init__(self):
        pass
    
    # Perform the attack
    def attack(self, dataset, model, eps, norm):
        # Split labels from images
        y = dataset.unbatch().map(lambda x, y: y)
    
        # create an adversarial dataset with FGSM
        x_adv = np.empty(shape=(0,64,64,3))
        for x_batch, y_batch in dataset:
            x_adv = np.concatenate((x_adv, fast_gradient_method(model, x_batch, eps=eps, norm=norm))) 
    
        # return pertubed dataset 
        return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x_adv),  y))

    # generate a dataset from the attack and evalute for used model (sanity checks)
    def generateDataset(self, folder, xy, model_substitute, eps=8, norm=np.inf):
        # the folder to store the dataset
        path = "datasets/adversarial_datasets/"+folder

        # Sanity check for the subsitute Model. Should be 100% when called with clean_dataset and substitute
        print("After removing wrong classified examples:")
        model_substitute.evaluate(xy)
        
        # After adversarial pertubation
        x_adversarial = self.attack(xy, model_substitute, eps, norm)
        print("On adversarial dataset:")
        model_substitute.evaluate(x_adversarial.batch(128))
        
        # Save dataset
        print("Saving dataset")
        x_adversarial = x_adversarial.batch(128)
        tf.data.experimental.save(x_adversarial, path)
        print("Saved")
