import neural_nets as p4
import data_pipeline as p1
import pandas as pd

if __name__ == '__main__':
    ######################## Logistic Regression #########################

    # Tuning the learning rate and epochs

    tuning_data = p4.breast_cancer_data.sample(frac=0.2)
    remaining_data = p4.breast_cancer_data.drop(index=tuning_data.index)

    label = 'class'
    # learning_rate = 0.1
    # epochs = 20

    folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    average_prediction = []

    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}, Tuning size: {}'.format(test.shape, train.shape, tuning_data.shape))
    #
    #     # train_encoded, test_encoded = p1.normalize(train, test)
    #     train_encoded, test_encoded = p1.normalize(train, tuning_data)
    #     # test_encoded = test
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(columns={2: 'benign', 4: 'malignant'})
    #     # test_class_encoded = p1.categorical_encoding(test[label], 1, None)
    #     # test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'})
    #     test_class_encoded = p1.categorical_encoding(tuning_data[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'}, errors='ignore')
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #     labels = ['benign', 'malignant']
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

    # Logistic Regression with Tuned parameters

    learning_rate = 0.09
    epochs = 20

    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: {}'.format(i))
        print('Test size: {}, Train size: {}'.format(test.shape, train.shape))

        train_encoded, test_encoded = p1.normalize(train, test)

        train_class_encoded = p1.categorical_encoding(train[label], 1, None)
        train_class_encoded = train_class_encoded.rename(columns={2: 'benign', 4: 'malignant'})
        test_class_encoded = p1.categorical_encoding(test[label], 1, None)
        test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'})

        train_encoded[train_class_encoded.columns] = train_class_encoded
        test_encoded[test_class_encoded.columns] = test_class_encoded

        train_encoded = train_encoded.drop(columns='class')
        test_encoded = test_encoded.drop(columns='class')
        labels = ['benign', 'malignant']
        train_std, test_std = train_encoded, test_encoded
        true_label_test = test_std[labels]
        test_std = test_std.drop(columns=true_label_test.columns)

        weights, unique_k = p4.logistic_regression(train_std, labels, learning_rate, epochs, True)

        pred = p4.predict(weights, unique_k, test_std)

        true_labels = true_label_test.idxmax(1)
        accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
        average_prediction.append(accuracy)
        print('Classification Accuracy on test: {:.3f}% for learning rate: {} and epochs: {}'.format(accuracy,
                                                                                                     learning_rate,
                                                                                                     epochs))
        print('*' * 50)
    print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))

    # Backpropagation Algorithm Tuning

    # label = 'class'
    # hidden_nodes_1 = 7
    # hidden_nodes_2 = 7
    # learning_rate = 0.05
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
    #     # train_encoded, test_encoded = p1.normalize(train, test)
    #     train_encoded, test_encoded = p1.normalize(train, tuning_data)
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(columns={2: 'benign', 4: 'malignant'}, errors='ignore')
    #     # test_class_encoded = p1.categorical_encoding(test[label], 1, None)
    #     # test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'})
    #     test_class_encoded = p1.categorical_encoding(tuning_data[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'}, errors='ignore')
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #     labels = ['benign', 'malignant']
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

    # Backpropagation with Tuned parameters

    # label = 'class'
    # hidden_nodes_1 = 4
    # hidden_nodes_2 = 7
    # learning_rate = 0.09
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
    #     train_encoded, test_encoded = p1.normalize(train, test)
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(columns={2: 'benign', 4: 'malignant'}, errors='ignore')
    #     test_class_encoded = p1.categorical_encoding(test[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'})
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #     labels = ['benign', 'malignant']
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

    # Tuning Parameters for Auto-encoding

    # hidden_nodes_1 = 7
    # hidden_nodes_2 = 3
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
    #     print('Test size: {}, Train size: {}, Tune size: {}'.format(test.shape, train.shape, tuning_data.shape))
    #
    #     # train_encoded, test_encoded = p1.normalize(train, test)
    #     train_encoded, test_encoded = p1.normalize(train, tuning_data)
    #
    #     train_class_encoded = p1.categorical_encoding(train[label], 1, None)
    #     train_class_encoded = train_class_encoded.rename(columns={2: 'benign', 4: 'malignant'}, errors='ignore')
    #     # test_class_encoded = p1.categorical_encoding(test[label], 1, None)
    #     # test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'})
    #     test_class_encoded = p1.categorical_encoding(tuning_data[label], 1, None)
    #     test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'}, errors='ignore')
    #
    #     train_encoded[train_class_encoded.columns] = train_class_encoded
    #     test_encoded[test_class_encoded.columns] = test_class_encoded
    #
    #     train_encoded = train_encoded.drop(columns='class')
    #     test_encoded = test_encoded.drop(columns='class')
    #     labels = ['benign', 'malignant']
    #     train_std, test_std = train_encoded, test_encoded
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #
    #     input_weights, hidden_weights_1, hidden_weights_2, unique_k = p4.autoencoder_class(train_std, labels,
    #                                                                                        hidden_nodes_1,
    #                                                                                        hidden_nodes_2,
    #                                                                                        learning_rate, epochs)
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

    # Autoencoder with Tuned Parameters

    hidden_nodes_1 = 5
    hidden_nodes_2 = 5
    learning_rate = 0.09
    epochs = 15

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

        train_encoded, test_encoded = p1.normalize(train, test)

        train_class_encoded = p1.categorical_encoding(train[label], 1, None)
        train_class_encoded = train_class_encoded.rename(columns={2: 'benign', 4: 'malignant'}, errors='ignore')
        test_class_encoded = p1.categorical_encoding(test[label], 1, None)
        test_class_encoded = test_class_encoded.rename(columns={2: 'benign', 4: 'malignant'})

        train_encoded[train_class_encoded.columns] = train_class_encoded
        test_encoded[test_class_encoded.columns] = test_class_encoded

        train_encoded = train_encoded.drop(columns='class')
        test_encoded = test_encoded.drop(columns='class')
        labels = ['benign', 'malignant']
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
