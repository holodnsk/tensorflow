# from tensorflow import keras as k # можно так
import keras as k # или так, keras является надстройкой над tensorflow
import numpy as np

# датасет входных данных в виде обычного pyton листа завернутый numpy массив, keras хорошо принимает только numpy массив
input_data = np.array([0.3, 0.7, 0.9])

#  датасет результатов которые нейронка будет учится предсказывать
output_data = np.array([0.5, 0.9, 1.0])

model = k.Sequential() # такой синтаксис. основная модель куда будем накручивать все слои,

# Dense  полносвязный многослойный перцептрон, линейная активация
model.add(k.layers.Dense(units=1, activation="linear")) # одна входящая переменная

# компилируем модель
# функция потерь mae (min absolete error), mse (средняя квадратичная ошибка)
# sgd - стохастический градиентный спуск.
# Лучше все параметры указывать явно, это говорит что я понимаю что происходит
model.compile(loss="mse", optimizer="sgd")

# здесь тренируется нейронка
fit_results = model.fit(x=input_data, y=output_data, epochs=100) # epochs=100 сколько  раз пройти весь датасет от начала до конца, на первой эпохе все веса случайные

# здесь смотрим как работает тренированная нейронка
predicted = model.predict([0.5])
print(predicted)