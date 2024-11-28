import os
import pickle

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# preparar dados
input_dir = "./clf-data"
categories = ['empty', 'not_empty']

data = []
labels = []

for category_ix, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_ix)

data = np.array(data)
labels = np.array(labels)


# treinar / testar split (separando os dados entre treino e teste)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


# classificador (treinando 12 classificadores)
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001],'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)


# teste de performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% de amostras foram classificadas corretamente.'.format(str(score*100)))

pickle.dump(best_estimator, open('./modelo.p', 'wb'))
