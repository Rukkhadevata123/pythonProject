import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import accuracy_score

# 读入文本(UTF-8)并移除换行符和回车符
data = open('text.txt', 'r', encoding='utf-8').read()
data = data.replace('\n', '').replace('\r', '')

# 字符去重
letters = list(set(data))
print(letters)
num_letters = len(letters)
print(num_letters)

# 建立字符和整数映射字典
int_to_char = {a: b for a, b in enumerate(letters)}
print(int_to_char)
char_to_int = {b: a for a, b in enumerate(letters)}
print(char_to_int)

# 从滑动窗口提取数据
def extract_data(_data: str, slide: int) -> list:
    x = []
    y = []
    for i in range(len(_data) - slide):
        x.append([a for a in _data[i:i + slide]])
        y.append(_data[i + slide])
    return x, y

# 字符到数字的批量转化
def char_to_int_Data(x: list, y: list, char_to_int: dict) -> list:
    x_to_int = []
    y_to_int = []
    for i in range(len(x)):
        x_to_int.append([char_to_int[a] for a in x[i]])
        y_to_int.append([char_to_int[a] for a in y[i]])
    return x_to_int, y_to_int

# 实现输入字符文章的批量处理
def data_processing(data: str, slide: int, num_letters: int, char_to_int: dict) -> tuple:
    char_Data = extract_data(data, slide)
    int_Data = char_to_int_Data(char_Data[0], char_Data[1], char_to_int)
    Input = int_Data[0]
    Output = list(np.array(int_Data[1]).flatten())

    Input_reshaped = np.array(Input).reshape(len(Input), slide)
    new = np.random.randint(0, 10, size=[Input_reshaped.shape[0], 
                                         Input_reshaped.shape[1], 
                                         num_letters])
    for i in range(Input_reshaped.shape[0]):        
        for j in range(Input_reshaped.shape[1]):
            new[i, j, :] = to_categorical(Input_reshaped[i, j], 
                                          num_classes=num_letters)
    return new, Output

# 使用前20个字符预测第21个字符
time_step = 20
X, y = data_processing(data, time_step, num_letters, char_to_int)
print(X.shape)
print(len(y))

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, len(y_train))

# 转换为独热码
y_train_category = to_categorical(y_train, num_classes=num_letters)
print(y_train_category)

# 模型建立
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=num_letters, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 调整批量大小和训练轮数
model.fit(X_train, y_train_category, batch_size=512, epochs=50, validation_split=0.1)

# 模型预测
y_test_predict = np.argmax(model.predict(X_test), axis=-1)
y_test_predict = list(map(int, y_test_predict))  # 转换为普通整数列表
accuracy_test = accuracy_score(y_test, y_test_predict)
print(accuracy_test)
print(y_test_predict)
print(y_test)

# 实际预测
new_letters = 'His success in the field of science was not due to his own '
new_letters = new_letters.lower()
new_letters = new_letters.replace('\n', '').replace('\r', '')
new_letters = [char_to_int[a] for a in new_letters]

# 预测多个字符
num_predict = 100  # 预测字符的数量
for _ in range(num_predict):
    input_seq = np.array(new_letters[-time_step:]).reshape(1, time_step)
    input_seq = to_categorical(input_seq, num_classes=num_letters)
    next_char = np.argmax(model.predict(input_seq), axis=-1)
    new_letters.append(next_char[0])

# 把实际预测的数字转换为字符
new_predict = [int_to_char[a] for a in new_letters]
print(''.join(new_predict))