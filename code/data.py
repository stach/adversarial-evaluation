# --- Dependencies

# Tensorflow
import tensorflow as tf

# NumPy
import numpy as np

# Other
import imageio
import time

# --- Tiny ImageNet Dataset loader

# the three loader methods are from Sonu Giri
# https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/ResNet_TinyImageNet.ipynb

class DataLoader:

    
    def __init__(self):
        self.PATH = "tiny-imagenet-200/"
        self.id_dict = self.get_id_dictionary()
        self.train_data, self.train_labels, self.test_data, self.test_labels = None, None, None, None

    # from https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/ResNet_TinyImageNet.ipynb
    def get_id_dictionary(self):
        id_dict = {}
        for i, line in enumerate(open( self.PATH + 'wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict
    
    # from https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/ResNet_TinyImageNet.ipynb
    def get_class_to_id_dict(self):
        id_dict = self.get_id_dictionary()
        all_classes = {}
        result = {}
        for i, line in enumerate(open( PATH + 'words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])      
        return result

    # from https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/ResNet_TinyImageNet.ipynb
    def get_data(self, id_dict):
        print('starting loading data')
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        t = time.time()
        for key, value in id_dict.items():
            train_data += [imageio.imread( self.PATH + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode="RGB") for i in range(500)]
            train_labels_ = np.array([[0]*200]*500)
            train_labels_[:, value] = 1
            train_labels += train_labels_.tolist()
            
        for line in open( self.PATH + 'val/val_annotations.txt'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append(imageio.imread( self.PATH + 'val/images/{}'.format(img_name), pilmode="RGB"))
            test_labels_ = np.array([[0]*200])
            test_labels_[0, id_dict[class_id]] = 1
            test_labels += test_labels_.tolist()

        print('finished loading data, in {} seconds'.format(time.time() - t))
        return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


    def saveData(self, batch_size=128):
        
        self.train_data, self.train_labels, self.test_data, self.test_labels = self.get_data(self.id_dict)
        
        test_data = self.test_data.astype("float32")
        train_data = self.train_data.astype("float32")

        shuffled_indices = np.random.permutation(train_data.shape[0])
        train_data_validate, train_data = train_data[shuffled_indices[:20000]], train_data[shuffled_indices[20000:]]
        train_labels_validate, train_labels = self.train_labels[shuffled_indices[:20000]], self.train_labels[shuffled_indices[20000:]]

        test_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(test_data),  tf.data.Dataset.from_tensor_slices(self.test_labels)))
            
        train_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_data),  tf.data.Dataset.from_tensor_slices(train_labels)))
        validate_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_data_validate),  tf.data.Dataset.from_tensor_slices(train_labels_validate)))

        test_dataset = test_dataset.batch(batch_size)
        train_dataset = train_dataset.batch(batch_size)
        validate_dataset = validate_dataset.batch(batch_size)
            
        tf.data.experimental.save(test_dataset, "datasets/tiny_imagenet/testing")
        tf.data.experimental.save(train_dataset, "datasets/tiny_imagenet/training")
        tf.data.experimental.save(validate_dataset, "datasets/tiny_imagenet/validation")
        
    # This method generates a clean dataset of size n for evaluation, 
    def generate_clean_evaluation_dataset(self, dataset, model, n=2500, path_evaluation_dataset = "datasets/evaluation_clean"):
        
        y = dataset.unbatch().map(lambda x, y: y)
        x = dataset.unbatch().map(lambda x, y: x)
        y = np.array(list(y))
        x = np.array(list(x))

        y_hat_class = np.argmax(model.predict(dataset), axis=1)
        y_class = np.argmax(y, axis=1) 
        filter_correct = (y_class==y_hat_class)

        x = x[filter_correct]
        y = y[filter_correct]
    
        N=min(sum(filter_correct), n)
    
        x = x[:N]
        y = y[:N]

        dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x),  tf.data.Dataset.from_tensor_slices(y)))
        dataset = dataset.batch(128)
        
        tf.data.experimental.save(dataset, path_evaluation_dataset)
