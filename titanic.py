import keras as k # на этапе подготовки данных лучше закоментировать, чтобы сборкой с керасом не тратило мое время
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_frame = pd.read_csv("titanic.csv")
input_names = ["Age","Sex","Pclass"] # имена колонок в исходной таблице
#print(data_frame[input_names]) # возможности pandas, проверить как читаются данные
#print(data_frame.head(n=5)) # возможности pandas
#print(data_frame["Age"]) # возможности pandas
#print(data_frame) # возможности pandas

output_names = ["Survived"]
#print(data_frame[output_names]) # возможности pandas, проверить как читаются данные



#print(data_frame["Age"].max())
#print(data_frame["Sex"].unique())
# есть вещественные данные (возраст), которые можно в одном столбце, а есть категориальные данные

###  нормализация данных
max_age = 100
# энкодер функция для подготовки данных из сырых данных, чтоб получилось как в "пример подготовки данных.jpg"
encoders = {"Age": lambda age: [age / max_age],
            "Sex": lambda gen: {"male": [0], "female": [1]}.get(gen),
            "Pclass": lambda pclass: {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}.get(pclass),
            "Survived": lambda s_value: [s_value]}

# это можно сделать через pandas
def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column]= values
    return result

# разделить на входные и выходные данные
def make_supervised(df):
    raw_input_data = data_frame[input_names]
    raw_output_data = data_frame[output_names]
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data)}



def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    #print(vectors)
    formated = []
    for vector_raw in list(zip(*vectors)): # будем итерировать все триплеты
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formated.append(vector)
    return formated



supervised = make_supervised(data_frame)
encoded_inputs = np.array(encode(supervised["inputs"]))
encoded_outputs = np.array(encode(supervised["outputs"]))
#print(encoded_inputs)
#print(encoded_outputs)
# [[0.22, 0, 0, 0, 1], [0.38, 1, 1, 0, 0], [0.26, 1, 0, 0, 1] ....] - лист входных переменных
# [[0], [1], [1] ....] - лист выходных переменных

###  окончена нормализация данных

# делим данные на три выборки
train_x = encoded_inputs[:600] # до 600-го элемента включительно
train_y = encoded_outputs[:600]

test_x = encoded_inputs[600:] # c 600-го элемента до конца (888 запись в titanic.csv)
test_y = encoded_outputs[600:]
# не делаем валидимрующую выборку

# модель нейронки
model = k.Sequential()

# добавим входной слой

# 5 переменных, rectified linear units линейный выпрямитель (если -1 или -10 будет 0, если  1 или 10 то будет 1)
model.add(k.layers.Dense(units=5, activation="relu"))

# не линейная функция активации сигмоид значения от 0 до единицы,
# впринципе это логарифмическая регрессия для вероятностных прогнозирований.
# В этом случае используется для задачи классификации
model.add(k.layers.Dense(units=1, activation="sigmoid"))
model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])

#  validation_split=0.2 80% данных пойдет на тренировку и 20% на валидацию
# verbose=0 если не хотим видеть все эпохи
fit_results = model.fit(x=train_x, y=train_y, epochs=100, validation_split=0.2)

#Epoch 1/100    loss: 0.2146 - accuracy: 0.7459 - val_loss: 0.2279 - val_accuracy: 0.6667
# ....
# Epoch 100/100 loss: 0.1923 - accuracy: 0.7616 - val_loss: 0.1951 - val_accuracy: 0.7583
# анализ вывода:
# хорошо: loss в конце меньше чем в начале
# хорошо: accuracy в конце больше чем в начале
# хорошо: val_loss и val_accuracy не сильно отличается от loss и accuracy

plt.plot(fit_results.history["loss"], label="Train")
plt.plot(fit_results.history["loss"], label="Train")

