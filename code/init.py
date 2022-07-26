# --- Dependencies

import os
import zipfile
import wget


# --- Init the Project


# Execute initially to install packages with pip; download dataset from source
class InitProject:

    def __init__(self):
        # The source for the Tiny ImageNet dataset
        self.URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def installRequirements(self):
        os.system("pip install -r requirements.txt")

    def retrieveDataset(self):
        # Retrieve tiny ImageNet from Stanford data source
        tiny_imagenet_200 = wget.download(self.URL)

        with zipfile.ZipFile("tiny-imagenet-200.zip", "r") as zip_ref:
            zip_ref.extractall(os.getcwd())

        # Remove zip
        os.remove("tiny-imagenet-200.zip")
