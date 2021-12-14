import keras
import sys
import h5py
import numpy as np

clean_data_filename = str(sys.argv[1])
poisoned_data_filename = str(sys.argv[2])
bd_model_filename = str(sys.argv[3])
prune_model_filename = str(sys.argv[4])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def G(bd_model, prune_model, bd_x_test, bd_y_test):
    bd_label_p = np.argmax(bd_model.predict(bd_x_test), axis=1)
    prune_label_p = np.argmax(prune_model.predict(bd_x_test), axis=1)
    attack_label = np.max(bd_y_test) + 1
    G_label = []

    for i in range(len(bd_y_test)):
        label = bd_label_p[i] if bd_label_p[i] == prune_label_p[i] else attack_label
        G_label.append(label)

    return np.array(G_label)


def main():
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    bd_model = keras.models.load_model(bd_model_filename)
    prune_model = keras.models.load_model(prune_model_filename)

    cl_label_p = np.argmax(prune_model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    print('Clean Classification accuracy:', clean_accuracy)

    bd_label_p = G(bd_model, prune_model, bd_x_test, bd_y_test)
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    print('Attack Success Rate:', asr)


if __name__ == '__main__':
    main()
