import os
import numpy as np

class Dataset:

    def __init__(self, path):
        self.paths = []
        self.images = []
        self.skeleton = []
        self.testes = []
        self.dataset_path = path

    def get_images_and_labels(self):
        print("Lendo dataset")
        for image_path in os.listdir(self.dataset_path):
            if image_path.endswith('.raw'):
                path = os.path.join(self.dataset_path, image_path)
                self.paths.append(path)

                path = open(path, 'rb')
                image = np.fromfile(path, dtype='uint8', count=300*300)
                image = image.reshape((300, 300))
                path.close()
                self.images.append(image)