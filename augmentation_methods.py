import numpy as np
import pywt


def threshold(coeffs):
    sigma = np.median(np.abs(coeffs - np.median(coeffs))) / 0.6745
    T = sigma * np.sqrt(2 * np.log(len(coeffs)))
    soft_thresholding = np.sign(coeffs) * np.maximum(np.abs(coeffs) - T, 0)
    return soft_thresholding


def swt_t(ecg_signal, wavelet, levels):
    augmented = np.zeros_like(ecg_signal)

    for lead in range(12):
        coeffs = pywt.swt(ecg_signal[:, lead], wavelet, level=levels)

        processed_coeffs = []
        for i in range(levels):
            A_c, D_c = coeffs[i]
            processed_D = threshold(D_c)
            processed_coeffs.append((A_c, processed_D))

        reconstructed = pywt.iswt(processed_coeffs, wavelet)
        augmented[:, lead] = reconstructed

    return augmented


def swt_augmentation(X, Y, class_names, minority_classes, samples_classes, wavelet, levels):
    X_augmented = []
    Y_augmented = []

    for class_name in minority_classes:
        class_idx = class_names.index(class_name)
        X_class = X[Y == class_idx]
        samples = samples_classes[class_name]

        for _ in range(samples):
            original_signal = X_class[np.random.randint(0, len(X_class))]
            augmented_signal = swt_t(original_signal, wavelet, levels)
            X_augmented.append(augmented_signal)
            Y_augmented.append(class_idx)

    return np.array(X_augmented), np.array(Y_augmented)


def window_slicing(X, Y, class_id, window_size, step_size):
    X_class = X[Y == class_id]

    X_slice = []
    Y_slice = []

    for signal in X_class:
        length = signal.shape[0]
        for start in range(0, length - window_size + 1, step_size):
            end = start + window_size
            X_slice.append(signal[start:end])
            Y_slice.append(class_id)

    return np.array(X_slice), np.array(Y_slice)


def window_slicing_augmantation(X, Y, class_names, window_size, step_size):
    X_augmented = []
    Y_augmented = []

    for class_id in np.unique(Y):
        s = step_size[class_id]
        X_slice, Y_slice = window_slicing(X, Y, class_id, window_size, s)

        X_augmented.append(X_slice)
        Y_augmented.append(Y_slice)

        print(f"Class {class_names[class_id]}: {len(X_slice)} samples")

    return np.concatenate(X_augmented), np.concatenate(Y_augmented)