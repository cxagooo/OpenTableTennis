from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=None):
    """
    将数据集分割为训练集、验证集和测试集。

    参数:
    - X: 特征数据 (numpy array, pandas DataFrame, etc.)
    - y: 标签数据 (numpy array, pandas Series, etc.)
    - test_size: 测试集所占的比例 (float, 默认为0.2)
    - val_size: 验证集所占的比例 (float, 默认为0.2)
    - random_state: 随机种子 (int, 默认为None)

    返回:
    - X_train: 训练集的特征数据
    - X_val: 验证集的特征数据
    - X_test: 测试集的特征数据
    - y_train: 训练集的标签数据
    - y_val: 验证集的标签数据
    - y_test: 测试集的标签数据
    """
    # 首先将数据集分割为训练集+验证集 (80%) 和 测试集 (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 再次将训练集+验证集分割为训练集 (80% of 80% = 64%) 和 验证集 (20% of 80% = 16%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size,
                                                      random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test
