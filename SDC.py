import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

test_size=0.3

trackdir = 'track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(trackdir, 'steering_log.csv'), names = columns)
pd.set_option('display.max_colwidth', -1)
data.head()

def path_leaf(dir_path):
  head, tail = ntpath.split(dir_path)
  return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

nbins = 25
bin_samples = 400
hist, bins = np.histogram(data['steering'], nbins)
center = (bins[:-1]+ bins[1:]) * 0.5

# remove data above threshold nbins to make histogram more uniform
print('total data:', len(data))
remove_list = []
for j in range(nbins):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_) 
  list_ = list_[bin_samples:]
  remove_list.extend(list_)

data.drop(data.index[remove_list], inplace=True)

def get_steering_angle(trackdir, df):
  image_dir = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_dir.append(os.path.join(trackdir, center.strip()))
    steering.append(float(indexed_data[3]))
    # left image append
    image_dir.append(os.path.join(trackdir,left.strip()))
    steering.append(float(indexed_data[3])+0.15)
    # right image append
    image_dir.append(os.path.join(trackdir,right.strip()))
    steering.append(float(indexed_data[3])-0.15)
  dir_list = np.asarray(image_dir)
  steering_angles = np.asarray(steering)
  return dir_list, steering_angles

dir_list, steering_angles = get_steering_angle(trackdir + '/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=test_size, random_state=6)

def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image

image = image_paths[random.randint(0, 1000)]
img_org = mpimg.imread(image)
img_zoomed = zoom(img_org)


def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

image = image_paths[random.randint(0, 1000)]
img_org = mpimg.imread(image)
img_panned = pan(img_org)

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

image = image_paths[random.randint(0, 1000)]
img_org = mpimg.imread(image)
img_brightened = img_random_brightness(img_org)

# if we flip the image we must flip the steering angle as well.
def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle

random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]

img_org = mpimg.imread(image)
img_flipped, steering_angle_flipped = img_random_flip(img_org, steering_angle)

# use imgaug data augmentation to improve the model's accuracy
def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)
    return image, steering_angle

for i in range(10):
  randnum = random.randint(0, len(image_paths) - 1)
  img_rand = image_paths[randnum]
  rand_steering = steerings[randnum]
  img_org = mpimg.imread(img_rand)
  img_augmented, steering = random_augment(img_rand, rand_steering)

def run_preprocess(img):
    img = img[60:135,:,:] # clip the unneeded parts of the image 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # apply some transforms
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
image = image_paths[100]
img_org = mpimg.imread(image)
img_preprocessed = run_preprocess(img_org)

def batch_generator(image_paths, steering_angle, batch_size, istraining):

  while True: # run continuously and stop with yield only
    batch_img = []
    batch_steering = []

    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)

      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_angle[random_index])

      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_angle[random_index]

      im = run_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))  

x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

  # use elu not relu 
  # relu doesnot work good
def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))

    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(100, activation = 'elu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation = 'elu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation = 'elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))   
    
## or use MiniVGGNet    
#    chanDim = -1
#    # first CONV => RELU => CONV => RELU => POOL layer set
#    model.add(Conv2D(32, (3, 3), padding = "same", input_shape=(66, 200, 3)))
#    model.add(Activation("elu"))
#    model.add(BatchNormalization(axis=chanDim))
#    model.add(Conv2D(32, (3, 3), padding="same"))
#    model.add(Activation("elu"))
#    model.add(BatchNormalization(axis=chanDim))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#    
#    # second CONV => RELU => CONV => RELU => POOL layer set
#    model.add(Conv2D(64, (3, 3), padding="same"))
#    model.add(Activation("elu"))
#    model.add(BatchNormalization(axis=chanDim))
#    model.add(Conv2D(64, (3, 3), padding="same"))
#    model.add(Activation("elu"))
#    model.add(BatchNormalization(axis=chanDim))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#    
#    # third CONV => RELU => CONV => RELU => POOL layer set
#    model.add(Conv2D(128, (3, 3), padding="same"))
#    model.add(Activation("elu"))
#    model.add(BatchNormalization(axis=chanDim))
#    model.add(Conv2D(128, (3, 3), padding="same"))
#    model.add(Activation("elu"))
#    model.add(BatchNormalization(axis=chanDim))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#    
#    # first (and only) set of FC => RELU layers
#    model.add(Flatten())
#    model.add(Dense(512))
#    model.add(Activation("elu"))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.5))
#    
#    # softmax classifier
#    model.add(Dense(1))
#    model.add(Activation("softmax"))   
    #  optimizer = Adam(lr=1e-4)
#    optimizer=keras.optimizers.RMSprop(lr=0.0001) #, rho=0.9, epsilon=None, decay=0.0)
    optimizer=keras.optimizers.Adadelta() #, epsilon=None, decay=0.0)
#  optimizer=keras.optimizers.Adagrad(lr=0.001) #, epsilon=None, decay=0.0)
#  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#  optimizer=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#  optimizer=keras.optimizers.Nadam(lr=0.0001) #, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='mse', optimizer=optimizer)    
    
    return model

model = nvidia_model()
print(model.summary())
  
train_img_gen_count=50
val_img_gen_count=50
train_nsteps=100
val_nsteps=100
lr1=0.0001;
  # bin_samples
  # Set callback functions to early stop training and save the best model so far
  #  checkpoint = [ModelCheckpoint(filepath='models.hdf5')]  
  # Train neural network
  #  history = network.fit(train_features, # Features
  #                      train_target, # Target vector
  #                      epochs=3, # Number of epochs
  #                      callbacks=checkpoint, # Checkpoint
  #                      verbose=0, # No output
  #                      batch_size=100, # Number of observations per batch
  #                      validation_data=(test_features, test_target)) # Data for evaluation  
  # augment 100 images 300 times in each epoch
  # model.fit requires entire data to be loaded into the memory
  # use model.fit_generator instead

save_dir = os.path.join(os.getcwd(), 'saved_models')
name2='_Adadelta_'+str(lr1)+'_'+str(ii*10)+'_'+str(bin_samples)+'_'+str(train_img_gen_count)+'_'+str(val_img_gen_count)+'_'+str(train_nsteps)+'_'+str(val_nsteps)+'_0.2_1.2_'+str(test_size)
model_name = 'model.{epoch:03d}'+name2+'.h5' 
if not os.path.isdir(save_dir):
  os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)  

#  def save_model(epoch):
#    if ((epoch % 10)==0):
#      save_dir = os.path.join(os.getcwd(), 'saved_models')
#      model_name = 'x_model.{epoch:03d}.h5' 
#      if not os.path.isdir(save_dir):
#        os.makedirs(save_dir)
#      filepath = os.path.join(save_dir, model_name)
#    return
#from keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
# Prepare model saving directory.
#save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)
#filepath = os.path.join(save_dir, model_name)
#
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=0,
                             save_best_only='FALSE',period=10)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
#if not data_augmentation:
#    print('Not using data augmentation.')
#    model.fit(x_train, y_train,
#              batch_size=batch_size,
#              epochs=epochs,
#              validation_data=(x_test, y_test),
#              shuffle=True,
#              callbacks=callbacks)

from keras.callbacks import CSVLogger  
#  name1='_adam_'+str(lr1)+'_'+str(ii*10)+'_'+str(bin_samples)+'_'+str(train_img_gen_count)+'_'+str(val_img_gen_count)+'_'+str(train_nsteps)+'_'+str(val_nsteps)+'_0.2_1.2_'+str(test_size)
csv_name= name2+'_log.csv'
  
csv_logger = CSVLogger(csv_name, append=False, separator=';')
#  history = model.fit(X_train, y_train,
#                      epochs=ii*10,
#                      validation_data    =(X_valid, y_valid),
#                      shuffle = 1,
#                      callbacks=[csv_logger, checkpoint ])
  
history = model.fit_generator(batch_generator(X_train, y_train, train_img_gen_count, 1),
                                    steps_per_epoch=train_nsteps, 
                                    epochs=20,
                                    validation_data    =batch_generator(X_valid, y_valid, val_img_gen_count, 0),
                                    validation_steps=val_nsteps,
                                    verbose=1,
                                    shuffle = 1,
                                    callbacks=[csv_logger, checkpoint ])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'], loc='upper left')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show