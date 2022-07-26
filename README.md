# Application of Transferable Adversarial Attacks on Convolutional Neuronal Networks: An Evaluation of Existing Attack and Defense Mechanisms

This project includes a digital version of the bachelor's thesis "Application of Transferable Adversarial Attacks on Convolutional Neural Networks: An Evaluation of Existing Attack and Defense Mechanisms" (Linus Stach, 2022) and the implementation of the conducted evaluation allowing the reproduction and validation of the results. To perform parts of the evaluation a GPU should be utilized. Likewise, the re-training of the generated models or the generation of adversarial datasets requires a lot of computing power. 

## Structure

```
└── adversarial-evaluation
    ├── code                                        
    │   ├── attack_bim.py                           
    │   ├── attack_fgsm.py                          
    │   ├── attack_momentum.py                      
    │   ├── attack_noise.py                         
    │   ├── data.py                                 
    │   ├── datasets                                
    │   │   ├── adversarial_datasets
    │   │   │   └── MI-FGSMAttack
    │   │   │       ├── e16
    │   │   │       ├── e32
    │   │   │       ├── e4
    │   │   │       └── e8
    │   │   ├── evaluation_clean
    │   │   ├── evaluation_noise
    │   │   └── tiny_imagenet
    │   │       ├── testing
    │   │       ├── training
    │   │       └── validation
    │   ├── Evaluation.ipynb
    │   ├── figures
    │   │   ├── ae_elephant.pdf
    │   │   ├── at.pdf
    │   │   ├── at_sampling.pdf
    │   │   ├── base.pdf
    │   │   ├── defensive_distillation.pdf
    │   │   ├── image_matrix_iii.pdf
    │   │   ├── image_matrix_ii.pdf
    │   │   ├── image_matrix_i.pdf
    │   │   ├── image_matrix_iv.pdf
    │   │   ├── image_matrix_v.pdf
    │   │   ├── madry.pdf
    │   │   ├── number_of_aml_papers.pdf
    │   │   └── superimposing_regularized.pdf
    │   ├── tiny-imagenet-200
    │   │   └── ...
    │   ├── init.py
    │   ├── Main.ipynb
    │   ├── requirements.txt
    │   ├── Results.ipynb
    │   ├── time.txt
    │   ├── trained_models
    │   │   └── defenses
    │   │       ├── adversarial_training
    │   │       ├── defensive_distillation
    │   │       ├── madry
    │   │       └── seiler
    │   ├── training_AdversarialTraining.py
    │   ├── training_DefensiveDistillation.py
    │   ├── training_Madry.py
    │   ├── training.py
    │   └── training_Seiler.py
    ├── README.md
    └── Thesis.pdf
 ```
 
 **Breakdown:**
 
* /tiny-imagenet-200/ contains the original tiny ImageNet dataset.
* /datasets/ contains all generated and used Tensorflow datasets (validation, adversarial, ...).
* /figures/ contains all generated graphics.
* "requirements.txt" contains all used libraries and their versions.
* "init<span>.py" initializes the project if libraries or tiny-imagenet-200 is missing.
* All modules "attack_*.py" implement an attack.
* All modules "training_*.py" train a model (with defense).
* "data<span>.py" transforms /tiny-imagenet-200/ into a format usable for the evaluation.
* "time.txt" stores the training time for the different models.
* The thesis is available in digital form in "Thesis.pdf".

 

## How to use

The notebooks "Main.ipynb" (1), "Evaluation.ipynb" (2) and "Results.ipynb" (3) are the active part of the evaluation and provide the possibility to interact:

1. Initializes the evaluation, generates and trains the various models, generates the datasets used in the evaluation.
2. Determines the accuracy values used in the black-box and white-box evaluation.
3. Creates several of the graphics used in the paper and saves them to /figures/... .


To reinitialize the evaluation and to download the dependencies, the following steps should be performed:

* Delete the folder /tiny-imagenet-200/.
* Run phase 0 of the notebook "Main.ipynb".

To re-train one or more models the following steps have to be performed:

* (Optional) Clear the content of the "time.txt" file.
* Delete the .h5 model in the respective folder of /trained_models/...
* Run phase 1 of the notebook "Main.ipynb".

If the to-be retrained model is the substitute, all evaluation datasets must be regenerated accordingly. These can generally be done as follows:

* Delete the corresponding folders in /datasets/... (evaluation_clean, evaluation_noise, adersarial_datasets/MI-FGSMAttack).
* Run phase 2 of the notebook "Main.ipynb".


    
    
