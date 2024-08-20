import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def __():
    from utils import list_change
    import pandas as pd
    import numpy as np
    import glob
    from keras import Sequential
    from keras.layers import LSTM, Dense
    import copy
    import matplotlib.pyplot as plt
    from utils import split_dataset, restore_changes
    import os
    import tensorflow as tf
    import cv2
    return (
        Dense,
        LSTM,
        Sequential,
        copy,
        cv2,
        glob,
        list_change,
        np,
        os,
        pd,
        plt,
        restore_changes,
        split_dataset,
        tf,
    )


@app.cell
def __(tf):
    # import tensorflow as tf
    tf.config.list_physical_devices('GPU')
    return


@app.cell
def __(np):
    def formating(X, Y):
        x_test = []
        y_test = []
        for x, y in zip(X, Y):
            x_test += x.tolist()
            y_test += y.tolist()
        return (np.array(x_test), np.array(y_test))
    return formating,


@app.cell
def __():
    # files = glob.glob('CutFrame_Output/output*/use2.txt')
    # data = [list_change(_f) for _f in files]
    return


@app.cell
def __(glob, list_change, np):
    data = []
    for _dir in glob.glob('CutFrame_Output/output*/'):
        data.append([list_change(_f) for _f in glob.glob(_dir + 'use*.txt')])
    data = np.array(data)
    return data,


@app.cell
def __(data):
    len(data)
    return


@app.cell
def __(data):
    data[0]
    return


@app.cell
def __():
    return


@app.cell
def __(copy, data, np):
    X = copy.deepcopy(data)
    Y = copy.deepcopy(data)
    X = np.delete(X, -1, axis=1)
    Y = np.delete(Y, 0, axis=1)
    return X, Y


@app.cell
def __(Y):
    Y.shape
    return


@app.cell
def __(X, Y, split_dataset):
    X, X_val, X_test, Y, Y_val, Y_test = split_dataset(X, Y)
    X_test = X_test[0]
    Y_test = Y_test[0]
    return X, X_test, X_val, Y, Y_test, Y_val


@app.cell
def __(X, cv2, restore_changes):
    index = 1
    _points = restore_changes(X[index], index, None)
    for _n, _f in enumerate(_points):
        _img = cv2.imread(f'./CutFrame_Output/output{index}/frame_{_n}.png')
        for _q, _p in enumerate(_f):
            cv2.circle(_img, (int(_p[0]), int(_p[1])), 3, (0, 255, 0), -1)
            cv2.putText(_img, str(_q), (int(_p[0]), int(_p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f'outputs/ori/output{_n}.png', _img)
    return index,


@app.cell
def __(X_test):
    X_test.shape
    return


@app.cell
def __(mo):
    mo.md("""\n    ### test\n""")
    return


@app.cell
def __(X, Y, np):
    X = np.array([X[0]])
    Y = np.array([Y[0]])
    X_test = X[0]
    Y_test = Y[0]
    return X, X_test, Y, Y_test


@app.cell
def __(Dense, LSTM, Sequential, X, X_test, Y, Y_test):
    l = []
    l0 = []
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[2], X.shape[3])))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    minimum_loss = float('inf')
    for epoch in range(2000):
        for x, y in zip(X, Y):
            loss = model.train_on_batch(x, y)
            l.append(loss)
        l0.append(model.evaluate(X_test, Y_test))
        if l0[-1] < minimum_loss:
            minimum_loss = l0[-1]
            model.save('best.h5')
    return epoch, l, l0, loss, minimum_loss, model, x, y


@app.cell
def __(l, l0, plt):
    plt.plot(l)
    plt.plot(l0)
    plt.show()
    return


@app.cell
def __(l0, plt):
    plt.plot(l0)
    return


@app.cell
def __(l, l0):
    print(f'best epoch: {min(l0)}', f'best batch: {min(l)}')
    return


@app.cell
def __(l0):
    l0[-1]
    return


@app.cell
def __(model):
    model.save('last.h5')
    return


@app.cell
def __(data):
    data[0]
    return


@app.cell
def __(data, restore_changes):
    restore_changes(data[0], 0, None)
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(X_test, Y_test, tf):
    model = tf.keras.models.load_model('best.h5')
    model.evaluate(X_test, Y_test)
    return model,


@app.cell
def __(data, model):
    a = model.predict(data[0]).tolist()
    return a,


@app.cell
def __(a):
    a[0]
    return


@app.cell
def __(data):
    data[0]
    return


@app.cell
def __(a, cv2, restore_changes):
    _points = restore_changes(a, 0, None)
    for _n, _f in enumerate(_points):
        _img = cv2.imread(f'./CutFrame_Output/output0/frame_{_n}.png')
        for _q, _p in enumerate(_f):
            cv2.circle(_img, (int(_p[0]), int(_p[1])), 3, (0, 255, 0), -1)
            cv2.putText(_img, str(_q), (int(_p[0]), int(_p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f'outputs/output{_n}.png', _img)
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
