# импорт библиотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# функция метрик
def metrics(y_test, y_test_predict, class_names):
    print(f'Accuracy Score test : {accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_test_predict, axis=1))}')
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_test_predict,  multi_class='ovr')}")
    print(f'F1-Score train : {f1_score(np.argmax(y_test, axis=1), np.argmax(y_test_predict, axis=1), average="macro")}')
    print(f'Precision Score test : {precision_score(np.argmax(y_test, axis=1), np.argmax(y_test_predict, axis=1), average="macro")}')
    print(f'Recall Score test : {recall_score(np.argmax(y_test, axis=1), np.argmax(y_test_predict, axis=1), average="macro")}')
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_test_predict, axis=1), target_names=class_names))

    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_test_predict, axis=1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix CNN')
    plt.show()

# функция загрузки данных
def load_data(path):
    X_train = np.load(path + 'X_train.npy')
    X_test = np.load(path + 'X_test.npy')
    X_val = np.load(path + 'X_val.npy')
    y_train = np.load(path + 'y_train.npy', allow_pickle=True)
    y_test = np.load(path + 'y_test.npy', allow_pickle=True)
    y_val = np.load(path + 'y_val.npy', allow_pickle=True)
    return X_train, X_test, X_val, y_train, y_test, y_val

# функция нормализации
def normalization(X_train, X_val, X_test):
    X_train_2d = X_train.reshape(-1, 12)
    X_val_2d = X_val.reshape(-1, 12)
    X_test_2d = X_test.reshape(-1, 12)

    scaler = StandardScaler()
    scaler.fit(X_train_2d)

    X_train_scaled_2d = scaler.transform(X_train_2d)
    X_val_scaled_2d = scaler.transform(X_val_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_2d.reshape(X_val.shape)
    X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled