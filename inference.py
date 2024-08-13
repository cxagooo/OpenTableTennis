import tensorflow as tf
from utils import restore_changes, list_change
import glob
import numpy as np


def process_data(input_file_path='input'):
    d = glob.glob(f'{input_file_path}/use*.txt')
    data = [list_change(f) for f in d]
    return data

def infer(model):
    data = process_data()
    y_hat = model.predict(data)
    del(data[0])
    del(y_hat[-1])
    return model(data)
