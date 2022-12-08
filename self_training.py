
# IMPORTS
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# IMAGE GENERATION
image_gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                               rescale=1/255, shear_range=0.1, zoom_range=0.4, fill_mode='nearest')

# BUILD THE MODEL
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3),  input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.6))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# TRAIN THE MODEL
train_image_gen = image_gen.flow_from_directory('self_trained_model/data/train',
                                                target_size=(150, 150), batch_size=16, class_mode='binary')
test_image_gen = image_gen.flow_from_directory('self_trained_model/data/test',
                                               target_size=(150, 150), batch_size=16, class_mode='binary')

results = model.fit(train_image_gen, epochs=20, steps_per_epoch=40, validation_data=test_image_gen, validation_steps=4)

# PLOT MODEL ACCURACY GRAPH
plt.plot(results.history['accuracy'], label='train_accuracy')
plt.plot(results.history['loss'], label='train_loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Report')
plt.legend()
plt.savefig('self_trained_model/reports/training_report.pdf')

# PLOT VALIDATION ACCURACY GRAPH
plt.plot(results.history['val_accuracy'], label='val_accuracy')
plt.plot(results.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Report')
plt.legend()
plt.savefig('self_trained_model/reports/validation_report.pdf')

# SAVE THE MODEL
model.summary()
model.save('self_trained_model/model/stop_cnn_model.h5')

# END OF CODE
