import numpy as np
import pandas as pd
import PIL.ImageOps

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, train_size = 7500, test_size = 2500)
X_train_scale = X_train/255.0
X_test_scale = X_test/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scale, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_pw = im_pil.convert('L')
    image_pw_resized = image_pw.resize((28, 28), Image.ANTIALIAS)

    pixel_filter = 20

    min_pixel = np.percentile(image_pw_resized, pixel_filter)
    image_pw_resized_inverted = np.clip(image_pw_resized-min_pixel, 0, 255)
    
    max_pixel = np.max(image_pw_resized)
    image_pw_resized_inverted = np.asarray(image_pw_resized_inverted)/max_pixel

    test_sample = np.array(image_pw_resized_inverted).reshape(1, 784)
    test_pred = clf.predict(test_sample)

    return test_pred[0]