import neural_nets as p4
import data_pipeline as p1
import pandas as pd

if __name__ == '__main__':
    ######################## Logistic Regression #########################

    tuning_data = p4.cars_data.sample(frac=0.2)
    remaining_data = p4.cars_data.drop(index=tuning_data.index)

    # Tuning the learning rate and epochs

    # label = 'acceptability'
    # learning_rate = 0.1
    # epochs = 15
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
    #     print('Test size: {}, Train size: {}, Tuning size: {}'.format(len(test), len(train), len(tuning_data)))
    #
    #     train_encoded = p1.categorical_encoding(dataframe=train, categorical_data_type=1, encoding_mapping=None)
    #     train_encoded = train_encoded.rename(
    #         columns={'acceptability_acc': 'acc', 'acceptability_good': 'good', 'acceptability_unacc': 'unacc',
    #                  'acceptability_vgood': 'vgood'}, errors='ignore')
    #
    #     test_encoded = p1.categorical_encoding(dataframe=tuning_data, categorical_data_type=1, encoding_mapping=None)
    #     test_encoded = test_encoded.rename(
    #         columns={'acceptability_acc': 'acc', 'acceptability_good': 'good', 'acceptability_unacc': 'unacc',
    #                  'acceptability_vgood': 'vgood'}, errors='ignore')
    #
    #     train_std, test_std = train_encoded, test_encoded
    #     labels = ['acc', 'good', 'unacc', 'vgood']
    #     # true_label_test = test_std.iloc[:, -label_encoded_columns:]
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #
    #     labels = ['acc', 'good', 'unacc', 'vgood']
    #     weights, unique_k = p4.logistic_regression(train_std, labels, 0.1, 30, True)
    #     pred = p4.predict(weights, unique_k, test_std)
    #
    #     true_labels = true_label_test.idxmax(1)
    #     accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
    #     average_prediction.append(accuracy)
    #     print('Classification Accuracy on test: {:.3f}% for eta: {}'.format(accuracy, 0.05))
    #     print('*' * 50)
    # print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))

    # Logistic Regression with Tuned Parameters

    # label = 'acceptability'
    # learning_rate = 0.1
    # epochs = 15
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
    #     print('Test size: {}, Train size: {}, Tuning size: {}'.format(len(test), len(train), len(tuning_data)))
    #
    #     train_encoded = p1.categorical_encoding(dataframe=train, categorical_data_type=1, encoding_mapping=None)
    #     train_encoded = train_encoded.rename(
    #         columns={'acceptability_acc': 'acc', 'acceptability_good': 'good', 'acceptability_unacc': 'unacc',
    #                  'acceptability_vgood': 'vgood'}, errors='ignore')
    #
    #     test_encoded = p1.categorical_encoding(dataframe=test, categorical_data_type=1, encoding_mapping=None)
    #     test_encoded = test_encoded.rename(
    #         columns={'acceptability_acc': 'acc', 'acceptability_good': 'good', 'acceptability_unacc': 'unacc',
    #                  'acceptability_vgood': 'vgood'}, errors='ignore')
    #
    #     train_std, test_std = train_encoded, test_encoded
    #     labels = ['acc', 'good', 'unacc', 'vgood']
    #     # true_label_test = test_std.iloc[:, -label_encoded_columns:]
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #
    #     labels = ['acc', 'good', 'unacc', 'vgood']
    #     weights, unique_k = p4.logistic_regression(train_std, labels, 0.1, 30, True)
    #     pred = p4.predict(weights, unique_k, test_std)
    #
    #     true_labels = true_label_test.idxmax(1)
    #     accuracy = p1.evaluation_metrics(true_output=true_labels, predicted_output=pred, metric_type=0)
    #     average_prediction.append(accuracy)
    #     print('Classification Accuracy on test: {:.3f}% for eta: {}'.format(accuracy, 0.05))
    #     print('*' * 50)
    # print('Average classification score across 5-folds: {:.3f}%'.format(p1.avg_output(average_prediction)))

    # Backpropagation Tuning

    # label = 'acceptability'
    # hidden_nodes_1 = 15
    # hidden_nodes_2 = 10
    # learning_rate = 0.1
    # epochs = 4
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
    #     print('Test size: {}, Train size: {}'.format(len(test), len(train)))
    #
    #     train_encoded = p1.categorical_encoding(dataframe=train, categorical_data_type=1, encoding_mapping=None)
    #     train_encoded = train_encoded.rename(
    #         columns={'acceptability_acc': 'acc', 'acceptability_good': 'good', 'acceptability_unacc': 'unacc',
    #                  'acceptability_vgood': 'vgood'})
    #
    #     test_encoded = p1.categorical_encoding(dataframe=tuning_data, categorical_data_type=1, encoding_mapping=None)
    #     test_encoded = test_encoded.rename(
    #         columns={'acceptability_acc': 'acc', 'acceptability_good': 'good', 'acceptability_unacc': 'unacc',
    #                  'acceptability_vgood': 'vgood'})
    #
    #     train_std, test_std = train_encoded, test_encoded
    #     labels = ['acc', 'good', 'unacc', 'vgood']
    #     true_label_test = test_std[labels]
    #     test_std = test_std.drop(columns=true_label_test.columns)
    #     print(test_std.shape)
    #
    #     labels = ['acc', 'good', 'unacc', 'vgood']
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
    # print('Average classification score across 5-folds: {:.3f}% for eta: {} at {} epochs'.format(
    #     p1.avg_output(average_prediction), learning_rate, epochs))

    # Backpropagation with Tuned parameters

    label = 'acceptability'
    hidden_nodes_1 = 15
    hidden_nodes_2 = 10
    learning_rate = 0.05
    epochs = 4

    folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=True, class_label=label)
    average_prediction = []

    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: {}'.format(i))
        print('Test size: {}, Train size: {}'.format(len(test), len(train)))

        train_encoded = p1.categorical_encoding(dataframe=train, categorical_data_type=1, encoding_mapping=None)
        train_encoded = train_encoded.rename(
            columns={'acceptability_acc': 'acc', 'acceptability_good': 'good', 'acceptability_unacc': 'unacc',
                     'acceptability_vgood': 'vgood'})

        test_encoded = p1.categorical_encoding(dataframe=test, categorical_data_type=1, encoding_mapping=None)
        test_encoded = test_encoded.rename(
            columns={'acceptability_acc': 'acc', 'acceptability_good': 'good', 'acceptability_unacc': 'unacc',
                     'acceptability_vgood': 'vgood'})

        train_std, test_std = train_encoded, test_encoded
        labels = ['acc', 'good', 'unacc', 'vgood']
        true_label_test = test_std[labels]
        test_std = test_std.drop(columns=true_label_test.columns)

        labels = ['acc', 'good', 'unacc', 'vgood']
        input_weights, hidden_weights_1, hidden_weights_2, unique_k = p4.backprop(train_std, labels, hidden_nodes_1,
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
    print('Average classification score across 5-folds: {:.3f}% for eta: {} at {} epochs'.format(
        p1.avg_output(average_prediction), learning_rate, epochs))
