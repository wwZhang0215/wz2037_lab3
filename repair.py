import keras
import keras.backend as K
import tensorflow as tf
import h5py
import numpy as np

def test():
    bd_model = keras.models.load_model('models/B_pi_net_0.1.h5')
    eval_data = h5py.File('data/valid.h5', 'r')
    x_data = np.array(eval_data['data'])
    y_data = np.array(eval_data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    print(cal_accuracy(bd_model, x_data, y_data))

def cal_accuracy(model, x_test, y_test):
    label_p = np.argmax(model.predict(x_test), axis=1)
    accuracy = np.mean(np.equal(label_p, y_test))
    return accuracy


def main(X):
    bd_model = keras.models.load_model('models/bd_net.h5')
    B_pi_model = tf.keras.models.clone_model(bd_model)

    # weights = h5py.File('models/bd_weights.h5', 'r')
    # print(weights.values())
    eval_data = h5py.File('data/valid.h5', 'r')
    x_data = np.array(eval_data['data'])
    y_data = np.array(eval_data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    extractor = keras.Model(inputs=bd_model.input, outputs=bd_model.get_layer('pool_3').output)
    features = extractor(x_data)
    # features.shape 11547*5*4*60
    feature_array = features.numpy()

    sum_by_channel = np.sum(feature_array, axis=(0, 1, 2))
    sorted_channel = np.argsort(sum_by_channel)

    weights = bd_model.get_weights()

    # names = [weight.name for layer in bd_model.layers for weight in layer.weights]

    # pool3_weights = np.array(weights['conv_3'])

    # print(names)
    # print(sum_by_channel.shape)
    # print(sorted_channel)
    # print(weights[4].shape)
    # print(weights[5])
    # print(sum_by_channel)

    clean_accuracy = cal_accuracy(bd_model, x_data, y_data)
    print('Clean Classification accuracy:', clean_accuracy)
    prune_accuracy = clean_accuracy

    output = 0

    for channel in sorted_channel:
        if sum_by_channel[channel] == 0:
            weights[5][channel] = 0
            continue
        weights[4][..., channel] = 0
        weights[5][channel] = 0
        B_pi_model.set_weights(weights)
        # print(B_pi_model.get_weights()[5])
        prune_accuracy = cal_accuracy(B_pi_model, x_data, y_data)
        print(prune_accuracy)
        if (clean_accuracy - prune_accuracy) > X[output]:
            file_name = "B_pi_net_" + str(X[output]) + ".h5"
            B_pi_model.save("models/"+file_name)
            print("file ", file_name, " output")
            output = output + 1
        if output >= len(X):
            break

    # while (clean_accuracy-prune_accuracy) < X:
    #     target_channel = np.arg


if __name__ == '__main__':
    main([0.02, 0.04, 0.10])
    # test()
