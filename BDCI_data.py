import numpy as np


if __name__ == '__main__':
    path = '/home/zqp/code/BDCI/train_data.npy'
    label = '/home/zqp/code/BDCI/train_label.npy'

    data = np.load(path)
    label = np.load(label)

    index = np.random.rand(data.shape[0])

    train_data = data[index>=0.2]
    train_label = label[index>=0.2]
    eval_data = data[index<0.2]
    eval_label = label[index<0.2]

    np.save('/home/zqp/code/BDCI/processed_data/train_data.npy', train_data[:, :2, ...])
    np.save('/home/zqp/code/BDCI/processed_data/train_label.npy', train_label)
    np.save('/home/zqp/code/BDCI/processed_data/eval_data.npy', eval_data[:, :2, ...])
    np.save('/home/zqp/code/BDCI/processed_data/eval_label.npy', eval_label)
    # print(label[0])