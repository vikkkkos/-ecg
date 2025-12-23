import augmentation_methods as aug
import model as md
import utils as ut
import numpy as np

class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
path = '/kaggle/input/augmentation-set/data/'

X_train, X_test, X_val, y_train, y_test, y_val = ut.load_data(path)
print("Данные загружены успешно!")
X_train, X_val, X_test = ut.normalization(X_train, X_val, X_test)
print("Нормализация прошла успешно!")

y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
steps = {
    0: 1000,
    1: 200,
    2: 200,
    3: 200,
    4: 100
}
X_train_ws, y_train_ws = aug.window_slicing_augmantation(X_train, y_train, class_names, window_size=600, step_size=steps)
minority_classes = ["MI", "STTC", "CD", "HYP"]
X_train_aug, y_train_aug = aug.swt_augmentation(X_train_ws, y_train_ws, class_names, minority_classes, 1000, wavelet='db5',
                                            levels=3)
X_train = np.concatenate([X_train_ws, np.array(X_train_aug)], axis=0)
y_train = np.concatenate([y_train_ws, np.array(y_train_aug)], axis=0)
print("Аугментация прошла успешно!")
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train_shuffled = X_train[indices]
y_train_shuffled = y_train[indices]
print("Сигналы перемешались")

# Параметры модели
input_shape = X_train[0].shape
classes = 5
batch_size = 64
epochs = 100
learning_rate = 0.001
patience = 10
model = md.ECG_Lense(input_shape, classes)
print("Модель инициализирована успешно!")
print("Начало обучения...")
model.fit(X_train_shuffled, y_train_shuffled, X_val, y_val, epochs, batch_size, learning_rate, patience, min_delta=1e-4,
          verbose=1)
print("Обучение прошло успешно!")
y_predict = model.predict(X_test, batch_size)
ut.metrics(y_test, y_predict, class_names)