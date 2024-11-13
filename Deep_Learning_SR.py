import matplotlib.pyplot as plt
import numpy
import numpy as np
import os

from keras import models, layers
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.layers import RandomFlip, RandomRotation, RandomZoom, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split


# Method to load the data
# Adding Labels to an image acording to the foldername
# Returns an array of images and labels
def load_data(data, img_size=(64, 64)):
    images = []
    labels = []
    label_names = ['Aanwijzingsborden', 'Gebodsborden', 'Gevaarsborden', 'Verbodsborden', 'Voorrangsborden']
    label_mapping = {label: i for i, label in enumerate(label_names)}
    for label in label_names:
        dir_label = os.path.join(data, label)
        for img in os.listdir(dir_label):
            img_path = os.path.join(dir_label, img)
            image = load_img(img_path, target_size=img_size, color_mode='grayscale')
            image = img_to_array(image)
            images.append(image)
            labels.append(label_mapping[label])

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def borden():
    # ######################### LOADING THE DATA ####################################################
    # Full dataset = 100 %.
    #   TRAINING set and VALIDATION set (80 %)
    #       Used during training :
    #           Training    -> using batches (subsets) from the training set.
    #           Validation  -> to evaluate the training process after an epoch (= iteration).
    #
    #   TEST set (20 %) is used after training, to test how well the trained model can handle new, unseen, data.

    print("\n------------------------------------------------->\nLOADING THE DATA SET.")

    images, labels = load_data('./borden', img_size=(64, 64))

    # Split loaded data into training, validation and test set
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels,
                                                                                        test_size=0.2, random_state=42)

    print('\n\ttrain_images shape :\t\t', train_images.shape)
    print('\ttrain_labels shape :\t\t', train_labels.shape)
    print('\tvalidation_images shape :\t', validation_images.shape)
    print('\tvalidation_labels shape :\t', validation_labels.shape)
    print('\ttest_images shape :\t\t\t', test_images.shape)
    print('\ttest_labels shape :\t\t\t', test_labels.shape)
    print()

    # A random number as example to see if everything is loaded correctly.
    image = train_images[5]
    plt.imshow(image, cmap='gray')
    plt.show()

    # ######################### PREPROCESSING #######################################################
    # 4 Steps:
    #       1. The data must be converted into a tensor
    #               https://hkilter.com/index.php?title=What_is_Tensor%3F
    #       2. The int values must be converted to floats.
    #       3. The dataset must be normalised to values between 0 and 1.
    #       4. The labels should be hot encoded.
    #               https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

    print("------------------------------------------------->\nPREPROCESSING.\n")

    train_images = train_images.astype('float32') / 255  # Convert to float + normalise.
    validation_images = validation_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    print('Shapes after reshaping into tensor :')
    print('\n\ttrain_images shape :\t\t', train_images.shape)
    print('\tvalidation_images shape :\t', validation_images.shape)
    print('\ttest_images shape :\t\t\t', test_images.shape)
    print()

    print('Hot encoding the labels.')
    train_labels = to_categorical(train_labels)  # Hot encoding: labels = [0..9].
    validation_labels = to_categorical(validation_labels)
    test_labels = to_categorical(test_labels)

    print('\nShapes of the labels after hot encoding :')
    print('\n\ttrain_labels shape :\t\t', train_labels.shape)
    print('\tvalidation_labels shape :\t', validation_labels.shape)
    print('\ttest_labels shape :\t\t\t', test_labels.shape)
    print('\nExample of hot encoded label: ', test_labels[0])

    print('\nPreprocessing done.')

    # ######################### MODELLING ###########################################################
    # Build the model.
    # This is a Deep Learning chain: input | hidden | output.

    print("\n------------------------------------------------->\nCONFIGURING THE NETWORK.\n")

    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # convert grayscale images to 3 channels
    train_images_rgb = np.repeat(train_images, 3, axis=-1)
    validation_images_rgb = np.repeat(validation_images, 3, axis=-1)
    test_images_rgb = np.repeat(test_images, 3, axis=-1)

    vgg16_base.trainable = False

    # model including the data augmentation
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(RandomFlip("horizontal_and_vertical"))
    model.add(RandomRotation(0.1))
    model.add(RandomZoom(0.1))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(vgg16_base)
    model.add(Flatten())
    model.add(layers.Dense(64, activation='relu', input_shape=[4096]))
    model.add(Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))

    print('\n------------------------------------------------->')
    print('MODEL SUMMARY.\n')
    print(model.summary())

    print('\n------------------------------------------------->\nCOMPILING THE NETWORK.\n')
    # Parameters of compile :
    #       loss        -> the type of loss function: values should decrease during training.
    #                   -> categorical_crossentropy = most appropriate for categorical multi-class classification.
    #       optimizer   -> method to update the weights of the neurons.
    #                   -> calculate derivative of the loss function.
    #       metrics     -> used to monitor the performance of the model.

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print('Modelling done.')

    # ######################### TRAINING ############################################################
    # Training the model.
    #   train_images        = input data.
    #   train_labels        = target data.
    #   epochs              = number of epochs to train the model.
    #   batch_size          = number of samples used for each batch in an epoch.
    #   verbose             = 1 -> one line per epoch.
    #   validation_data     = data on which to evaluate the loss at the end of each epoch.

    print('\n------------------------------------------------->\nTRAINING THE NETWORK.\n')

    learning_evolution = model.fit(train_images_rgb,
                                   train_labels,
                                   epochs=100,
                                   batch_size=128,
                                   verbose=1,
                                   validation_data=(validation_images_rgb, validation_labels))

    print('Training done.')

    # ######################### TESTING #############################################################
    # Evaluate the model on the test data.

    print('\n------------------------------------------------->\nTESTING THE NETWORK.\n')

    test_loss, test_accuracy = model.evaluate(test_images_rgb, test_labels)
    print('\nTest Loss:\t\t', round(test_loss, 3))
    print('Test Accuracy:\t', round(test_accuracy, 3))

    # ######################### MAKING PREDICTIONS ##################################################

    print('\n------------------------------------------------->\nUSING THE MODEL, MAKING PREDICTIONS.\n')

    # Get predicted labels.
    predict_x = model.predict(test_images_rgb)
    classes_x = numpy.argmax(predict_x, axis=1)

    print('\nPredictions of the first 10 test images:\n')

    for i in range(10):
        print("Target = %s -- Corresponds to %s -- Predicted %s" % (
            test_labels[i], list(test_labels[i]).index(1), classes_x[i]))

    ########################## TESTING LEARNING BEHAVIOR ########################################
    # Show plot to detect overfitting, using history dictionary.

    accuracy = learning_evolution.history['accuracy']
    val_accuracy = learning_evolution.history['val_accuracy']
    loss = learning_evolution.history['loss']
    val_loss = learning_evolution.history['val_loss']

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    borden()

