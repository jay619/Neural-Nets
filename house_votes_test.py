import pandas as pd

import data_pipeline as p1
import neural_nets as p4

if __name__ == '__main__':
    ######################## Logistic Regression #########################

    # Tuning the learning rate and epochs

    tuning_data = p4.house_votes_data.sample(frac=0.2)
    remaining_data = p4.house_votes_data.drop(index=tuning_data.index)

    # label = 'class'
    # learning_rate = 0.03
    # epochs = 20
    #
    # folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    # average_prediction = []
    #
    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}, Tuning size: {}'.format(test.shape, train.shape, tuning_data.shape))
    #
    #     train_encoded = train
    #     test_encoded = tuning_data
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'}, errors='ignore')
    #     test_class_encoded = p1.categorical_encoding(tuning_data[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'}, errors='ignore')
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #
    #     labels = ['democrat', 'republican']
    #     train_std, test_std = train_encoded, test_encoded
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #
    #     weights, unique_k = p4.logistic_regression(train_std, labels, learning_rate, epochs, True)
    #
    #     pred = p4.predict(weights, unique_k, test_std)
    #
    #     true_labels = true_label_test.idxmax(1)
    #     accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
    #     average_prediction.append(accuracy)
    #     print('Classification Accuracy on test: {:.3f}% for learning rate: {} and epochs: {}'.format(accuracy,
    #                                                                                                  learning_rate,
    #                                                                                                  epochs))
    #     print('*' * 50)
    # print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))

    # Logistic Regression with Tuned Parameters

    # label = 'class'
    # learning_rate = 0.03
    # epochs = 20
    #
    # folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    # average_prediction = []
    #
    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}'.format(test.shape, train.shape))
    #
    #     train_encoded = train
    #     test_encoded = test
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'}, errors='ignore')
    #     test_class_encoded = p1.categorical_encoding(test[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'}, errors='ignore')
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #
    #     labels = ['democrat', 'republican']
    #     train_std, test_std = train_encoded, test_encoded
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #
    #     weights, unique_k = p4.logistic_regression(train_std, labels, learning_rate, epochs, True)
    #
    #     pred = p4.predict(weights, unique_k, test_std)
    #
    #     true_labels = true_label_test.idxmax(1)
    #     accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
    #     average_prediction.append(accuracy)
    #     print('Classification Accuracy on test: {:.3f}% for learning rate: {} and epochs: {}'.format(accuracy,
    #                                                                                                  learning_rate,
    #                                                                                                  epochs))
    #     print('*' * 50)
    # print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))

    # Backpropogation Tuning

    # label = 'class'
    # hidden_nodes_1 = 35
    # hidden_nodes_2 = 15
    # learning_rate = 0.03
    # epochs = 10
    #
    # folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    # average_prediction = []
    #
    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}'.format(test.shape, train.shape))
    #
    #     train_encoded = train
    #     test_encoded = tuning_data
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'})
    #     test_class_encoded = p1.categorical_encoding(tuning_data[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'})
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #     print('Features: ', test_encoded.shape)
    #     labels = ['democrat', 'republican']
    #
    #     train_std, test_std = train_encoded, test_encoded
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #
    # input_weights, hidden_weights_1, hidden_weights_2, unique_k = p4.backprop(train_std, labels, hidden_nodes_1,
    # hidden_nodes_2, learning_rate, epochs) pred = p4.predict_backprop(test_std, unique_k, hidden_nodes_1,
    # hidden_nodes_2, input_weights, hidden_weights_1, hidden_weights_2)
    #
    #     true_labels = true_label_test.idxmax(1)
    #     accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
    #     average_prediction.append(accuracy)
    #     print('Classification Accuracy on test: {:.3f}% for learning rate: {} and epochs: {} with {} hidden neurons '
    #           'at level 1 and {} hidden neurons at level 2'.format(accuracy,
    #                                                                learning_rate,
    #                                                                epochs,
    #                                                                hidden_nodes_1,
    #                                                                hidden_nodes_2))
    #     print('*' * 50)
    # print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))
    # print('-' * 50)

    # Backpropagation with Tuned Parameters
    #
    # label = 'class'
    # hidden_nodes_1 = 35
    # hidden_nodes_2 = 15
    # learning_rate = 0.03
    # epochs = 10
    #
    # folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    # average_prediction = []
    #
    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}'.format(test.shape, train.shape))
    #
    #     train_encoded = train
    #     test_encoded = test
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'})
    #     test_class_encoded = p1.categorical_encoding(test[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'})
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #
    #     labels = ['democrat', 'republican']
    #
    #     train_std, test_std = train_encoded, test_encoded
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #
    #     input_weights, hidden_weights_1, hidden_weights_2, unique_k = p4.backprop(train_std, labels, hidden_nodes_1,
    #                                                                               hidden_nodes_2, learning_rate, epochs)
    #     pred = p4.predict_backprop(test_std, unique_k, hidden_nodes_1, hidden_nodes_2, input_weights, hidden_weights_1,
    #                                hidden_weights_2)
    #
    #     true_labels = true_label_test.idxmax(1)
    #     accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
    #     average_prediction.append(accuracy)
    #     print('Classification Accuracy on test: {:.3f}% for learning rate: {} and epochs: {} with {} hidden neurons '
    #           'at level 1 and {} hidden neurons at level 2'.format(accuracy,
    #                                                                learning_rate,
    #                                                                epochs,
    #                                                                hidden_nodes_1,
    #                                                                hidden_nodes_2))
    #     print('*' * 50)
    # print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))

    # Autoencoder Parameter Tuning

    # label = 'class'
    # hidden_nodes_1 = 30
    # hidden_nodes_2 = 12
    # learning_rate = 0.05
    # epochs = 50
    #
    # folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    # average_prediction = []
    #
    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}'.format(test.shape, train.shape))
    #
    #     train_encoded = train
    #     test_encoded = tuning_data
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'})
    #     test_class_encoded = p1.categorical_encoding(tuning_data[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(
    #         columns={'class_democrat': 'democrat', 'class_republican': 'republican'})
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #     print('Features: ', test_encoded.shape)
    #     labels = ['democrat', 'republican']
    #
    #     train_std, test_std = train_encoded, test_encoded
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #
    #     input_weights, hidden_weights_1, hidden_weights_2, unique_k = p4.autoencoder_class(train_std, labels,
    #                                                                                        hidden_nodes_1,
    #                                                                                        hidden_nodes_2, learning_rate,
    #                                                                                        epochs)
    #     pred = p4.predict_backprop(test_std, unique_k, hidden_nodes_1,
    #                                hidden_nodes_2, input_weights, hidden_weights_1, hidden_weights_2)
    #
    #     true_labels = true_label_test.idxmax(1)
    #     accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
    #     average_prediction.append(accuracy)
    #     print('Classification Accuracy on test: {:.3f}% for learning rate: {} and epochs: {} with {} hidden neurons '
    #           'at level 1 and {} hidden neurons at level 2'.format(accuracy,
    #                                                                learning_rate,
    #                                                                epochs,
    #                                                                hidden_nodes_1,
    #                                                                hidden_nodes_2))
    #     print('*' * 50)
    # print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))

    # Autoencoder with Tuned Parameters

    label = 'class'
    hidden_nodes_1 = 30
    hidden_nodes_2 = 12
    learning_rate = 0.05
    epochs = 25

    folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    average_prediction = []

    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: {}'.format(i))
        print('Test size: {}, Train size: {}'.format(test.shape, train.shape))

        train_encoded = train
        test_encoded = test

        train_class_encoded = p1.categorical_encoding(train[label], 1, None)
        train_class_encoded = train_class_encoded.rename(
            columns={'class_democrat': 'democrat', 'class_republican': 'republican'})
        test_class_encoded = p1.categorical_encoding(test[label], 1, None)
        test_class_encoded = test_class_encoded.rename(
            columns={'class_democrat': 'democrat', 'class_republican': 'republican'})

        train_encoded[train_class_encoded.columns] = train_class_encoded
        test_encoded[test_class_encoded.columns] = test_class_encoded

        train_encoded = train_encoded.drop(columns='class')
        test_encoded = test_encoded.drop(columns='class')

        labels = ['democrat', 'republican']

        train_std, test_std = train_encoded, test_encoded
        true_label_test = test_std[labels]
        test_std = test_std.drop(columns=true_label_test.columns)

        input_weights, hidden_weights_1, hidden_weights_2, unique_k = p4.autoencoder_class(train_std, labels, hidden_nodes_1,
                                                                                  hidden_nodes_2, learning_rate, epochs)
        pred = p4.predict_backprop(test_std, unique_k, hidden_nodes_1, hidden_nodes_2, input_weights, hidden_weights_1,
                                   hidden_weights_2)

        true_labels = true_label_test.idxmax(1)
        accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
        average_prediction.append(accuracy)
        print('Classification Accuracy on test: {:.3f}% for learning rate: {} and epochs: {} with {} hidden neurons '
              'at level 1 and {} hidden neurons at level 2'.format(accuracy,
                                                                   learning_rate,
                                                                   epochs,
                                                                   hidden_nodes_1,
                                                                   hidden_nodes_2))
        print('*' * 50)
    print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))
