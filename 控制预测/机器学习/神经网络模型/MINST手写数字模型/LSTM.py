from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
# 无视以上警告，能正常运行
from Load_Data import X_train, y_train, X_test, y_test

""" # 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data() """

# 数据预处理
X_train = X_train.reshape(-1, 28*28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28*28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 建立LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(28*28, 1)))  # 更新输入形状
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)