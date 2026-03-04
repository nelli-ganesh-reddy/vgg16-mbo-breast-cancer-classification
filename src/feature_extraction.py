import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)


def extract_features(folder):

    features_list = []
    labels = []
    filenames = []

    for label_name in ['benign', 'malignant']:
        label_folder = os.path.join(folder, label_name)
        label_value = 0 if label_name == 'benign' else 1

        for img_name in os.listdir(label_folder):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_folder, img_name)

                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                features = model.predict(x, verbose=0).flatten()

                features_list.append(features)
                labels.append(label_value)
                filenames.append(img_name)

    df = pd.DataFrame(features_list)
    df.insert(0, "filename", filenames)
    df.insert(1, "label", labels)

    return df