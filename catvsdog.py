import numpy as np
import datasets
#import matplotlib.pyplot as plt
import tensorflow as tf

### Load dataset
whole_data = datasets.load_dataset('cats_vs_dogs', split = 'train')

### Train and test spliting
train_and_test_data = whole_data.train_test_split(test_size = 0.2)
train_images, train_labels = train_and_test_data['train'].select_columns('image'), train_and_test_data['train'].select_columns('labels')
test_images, test_labels = train_and_test_data['test'].select_columns('image'), train_and_test_data['test'].select_columns('labels')

train_labels = np.array(train_labels['labels'])
test_labels = np.array(test_labels['labels'])

### Image processing
def transforms(examples):
    examples['img'] = [image.convert("RGB").resize((28, 28)) for image in examples['image']]
    return examples

train_images = train_images.map(transforms, remove_columns = ['image'], batched = True)
test_images = test_images.map(transforms, remove_columns = ['image'], batched = True)

train_tensor = []
test_tensor = []

for index in range(len(train_images)):
    train_tensor.append(np.array(train_images[index]['img']))

for index in range(len(test_images)):
    test_tensor.append(np.array(test_images[index]['img']))

train_tensor = np.array(train_tensor)
test_tensor = np.array(test_tensor)

### Normalizing
train_tensor = train_tensor / 255
test_tensor = test_tensor / 255

### Class names of data
class_names = ['Cat', 'Dog']

### CNN
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (28, 28, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(2, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_tensor, train_labels, epochs = 15)

model.save("catvsdog.model")

loss, accuracy = model.evaluate(test_tensor, test_labels)
print(f'Accuracy: {accuracy * 100}\nLoss: {loss}')
"""
