from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import np_utils
from keras import losses
from keras.datasets import cifar100
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

(x_train,y_train),(x_test,y_test)=cifar100.load_data()

#ödev için gerekli sınıfları ayırmak için yazılan kodlar
index = np.where(y_train == 23)
X_train = x_train[index[0]]
Y_train = y_train[index[0]]

index1 = np.where(y_train == 24)
X1_train = x_train[index1[0]]
Y1_train = y_train[index1[0]]

index2 = np.where(y_train == 37)
X2_train = x_train[index2[0]]
Y2_train = y_train[index2[0]]

index3 = np.where(y_train == 40)
X3_train = x_train[index3[0]]
Y3_train = y_train[index3[0]]

index4 = np.where(y_train == 80)
X4_train = x_train[index4[0]]
Y4_train = y_train[index4[0]]
#
index5 = np.where(y_test == 23)
X_test = x_test[index5[0]]
Y_test = y_test[index5[0]]

index6 = np.where(y_test == 24)
X1_test = x_test[index6[0]]
Y1_test = y_test[index6[0]]

index7 = np.where(y_test == 37)
X2_test = x_test[index7[0]]
Y2_test = y_test[index7[0]]

index8 = np.where(y_test == 40)
X3_test = x_test[index8[0]]
Y3_test = y_test[index8[0]]

index9= np.where(y_test == 80)
X4_test = x_test[index9[0]]
Y4_test = y_test[index9[0]]

r=np.concatenate((X_train, X1_train), axis=0)
r=np.concatenate((r, X2_train), axis=0)
r=np.concatenate((r, X3_train), axis=0)
r=np.concatenate((r, X4_train), axis=0)

r2=np.concatenate((Y_train, Y1_train), axis=0)
r2=np.concatenate((r2, Y2_train), axis=0)
r2=np.concatenate((r2, Y3_train), axis=0)
r2=np.concatenate((r2, Y4_train), axis=0)

r3=np.concatenate((X_test, X1_test), axis=0)
r3=np.concatenate((r3, X2_test), axis=0)
r3=np.concatenate((r3, X3_test), axis=0)
r3=np.concatenate((r3, X4_test), axis=0)

r4=np.concatenate((Y_test, Y1_test), axis=0)
r4=np.concatenate((r4, Y2_test), axis=0)
r4=np.concatenate((r4, Y3_test), axis=0)
r4=np.concatenate((r4, Y4_test), axis=0)


x_train=r
y_train=r2
x_test=r3
y_test=r4


x_train=x_train / 255.0
x_test=x_test/255.0

yedek=y_train

yedek=np.where(yedek==23, 1, yedek)
yedek=np.where(yedek==24, 2, yedek)
yedek=np.where(yedek==37, 3, yedek)
yedek=np.where(yedek==40, 4, yedek)
yedek=np.where(yedek==80, 5, yedek)

yedek_c = to_categorical(yedek)
yedek_c=np.delete(yedek_c,0,1)
y_train=yedek_c

y_test=np.where(y_test==23,1,y_test)
y_test=np.where(y_test==24,2,y_test)
y_test=np.where(y_test==37,3,y_test)
y_test=np.where(y_test==40,4,y_test)
y_test=np.where(y_test==80,5,y_test)
y_test = to_categorical(y_test)
y_test=np.delete(y_test,0,1)

x_train, y_train = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)

#eğitim için kurulan model fonksiyonu
def model1():
    model=models.Sequential()
    model.add(layers.Conv2D(32,
                            (3,3),
                            activation='relu',
                            padding='same',
                            input_shape= (32, 32, 3)))
    #model.add(Activation('relu'))
    model.add(layers.Conv2D(64,
                            (3,3),
                            padding='same',
                            activation='relu'))
    model.add(layers.MaxPool2D())

    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128,
                            (3,3),
                            padding='same',
                            activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128,
                            (3,3),
                            padding='same',
                            activation='relu'))
    model.add(layers.MaxPool2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(5,activation='softmax'))
    model.summary()
    #modeli derle

    from keras import optimizers
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['acc'])

    return model


#modeli derleme işlemı
model=model1()
 

def fonk(img):
    plt.imshow(img)
    plt.show()
    return img

from keras.preprocessing.image import ImageDataGenerator

#veri zenginleştirme işlemleri
train_datagen = ImageDataGenerator( 
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=25)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow(
        x_test,
        y_test,
        batch_size=25)

history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=500,
      validation_data=validation_generator,
      validation_steps=50)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

import numpy as np

np.save('history.npy',(acc,val_acc,loss,val_loss))
(acc,val_acc,loss,val_loss)=np.load('history.npy')

epochs = range(1,501)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

#

epochs = range(1,501)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Egitim ve Validation Dogruluk Grafigi')
plt.xlabel('Epoch Sayisi')
plt.ylabel('Dogruluk Orani')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'blue', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Eğitim ve Validation Kayıp Grafigi')
plt.xlabel('Epoch Sayisi')
plt.ylabel('Loss Miktari')
plt.legend()

plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)

f, axarr = plt.subplots(5,5)
axarr[0,0].imshow(X_train[0])
axarr[0,1].imshow(X_train[1])
axarr[0,2].imshow(X_train[2])
axarr[0,3].imshow(X_train[3])
axarr[0,4].imshow(X_train[4])

axarr[1,0].imshow(X1_train[0])
axarr[1,1].imshow(X1_train[1])
axarr[1,2].imshow(X1_train[2])
axarr[1,3].imshow(X1_train[3])
axarr[1,4].imshow(X1_train[4])

axarr[2,0].imshow(X2_train[0])
axarr[2,1].imshow(X2_train[1])
axarr[2,2].imshow(X2_train[2])
axarr[2,3].imshow(X2_train[3])
axarr[2,4].imshow(X2_train[4])

axarr[3,0].imshow(X3_train[0])
axarr[3,1].imshow(X3_train[1])
axarr[3,2].imshow(X3_train[2])
axarr[3,3].imshow(X3_train[3])
axarr[3,4].imshow(X3_train[4])

axarr[4,0].imshow(X4_train[0])
axarr[4,1].imshow(X4_train[1])
axarr[4,2].imshow(X4_train[2])
axarr[4,3].imshow(X4_train[3])
axarr[4,4].imshow(X4_train[4])

plt.show()

y_pred=model.predict(x_test)
from sklearn.metrics import confusion_matrix
import seaborn as sn
#from sklearn.metrics import plot_confusion_matrix
y_pred1=np.argmax(y_pred, axis=1)
y_test1=np.argmax(y_test, axis=1)
cm=confusion_matrix(y_test1,y_pred1)


df_cm = pd.DataFrame(cm, index=["cloud", "cockroach", "house ", "lamp","squirrel"], columns=["cloud", "cockroach", "house ", "lamp","squirrel"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,cmap="Blues")

single_test = model.predict(np.expand_dims(x_test[19], axis=0))

x_test1=x_test*255
