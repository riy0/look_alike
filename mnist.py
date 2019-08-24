from keras.datasets import mnist                    #data load
from keras.utils.np_utils import to_categorical     #preprocessing
from keras.models import Sequential                 #construct model
from keras.layers import Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')/255

x_test= x_test.reshape(10000, 784)
x_test= x_test.astype('float32')/255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=12, verbose=1)

score = model.evaluate(x_test, y_test)
print(score[0])
print(score[1])
 

