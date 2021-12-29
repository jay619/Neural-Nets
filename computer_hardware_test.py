import pandas as pd

import data_pipeline as p1
import neural_nets as p4

if __name__ == '__main__':

    ######################## Logistic Regression #########################
    label = 'prp'
    tuning_data = p4.comp_hardware_data.sample(frac=0.2)
    remaining_data = p4.comp_hardware_data.drop(index=tuning_data.index)

    # learning_rate = 0.05
    # epochs = 10

    # Tuning Parameters

    # folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=False, class_label=label)
    # average_prediction = []
    #
    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}, Tune size: {}'.format(len(test), len(train), len(tuning_data)))
    #
    #     train_std, test_std = p1.normalize(train=train, test=tuning_data)
    #
    #     true_output = test_std[label].to_numpy().reshape(len(test_std), 1)
    #     # Just evaluating the original tree on the test data for comparison
    #     test_std = test_std.drop(columns=[label])
    #     weights = p4.linear_regression(training_data=train_std, output_label=label, eta=0.1, max_iter=20)
    #     pred = p4.predict_output_simple_reg(test_std, weights)
    #     accuracy = p1.evaluation_metrics(true_output=true_output, predicted_output=pred, metric_type=1)
    #     average_prediction.append(accuracy)
    #     print('Mean Squared Error on Tuning data: {:.3f}'.format(accuracy))
    #     print('*' * 50)
    # print('Average Mean Squared Error across 5-folds: {:.3f}'.format(p1.avg_output(average_prediction)))
    # print('-' * 50)

    # Linear Regression with Tuned Parameters

    # learning_rate = 0.1
    # epochs = 10
    #
    # folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=False, class_label=label)
    # average_prediction = []
    #
    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}'.format(len(test), len(train), len(tuning_data)))
    #
    #     train_std, test_std = p1.normalize(train=train, test=test)
    #
    #     true_output = test_std[label].to_numpy().reshape(len(test_std), 1)
    #     # Just evaluating the original tree on the test data for comparison
    #     test_std = test_std.drop(columns=[label])
    #     weights = p4.linear_regression(training_data=train_std, output_label=label, eta=0.1, max_iter=20)
    #     pred = p4.predict_output_simple_reg(test_std, weights)
    #     accuracy = p1.evaluation_metrics(true_output=true_output, predicted_output=pred, metric_type=1)
    #     average_prediction.append(accuracy)
    #     print('Mean Squared Error on Tuning data: {:.3f}'.format(accuracy))
    #     print('*' * 50)
    # print('Average Mean Squared Error across 5-folds: {:.3f}'.format(p1.avg_output(average_prediction)))
    # print('-' * 50)

    # Backpropagation Tuning

    # hidden_nodes_1 = 5
    # hidden_nodes_2 = 2
    # learning_rate = 0.05
    # epochs = 250
    #
    # folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=False, class_label=label)
    # average_prediction = []
    #
    # for i, x in enumerate(folds):
    #     test = x
    #     train_set = [l for idx, l in enumerate(folds) if idx != i]
    #     train = pd.DataFrame()
    #     for val in train_set:
    #         train = train.append(val)
    #     print('Running fold: {}'.format(i))
    #     print('Test size: {}, Train size: {}, Tune size: {}'.format(len(test), len(train), len(tuning_data)))
    #
    #     # print(train_encoded.columns)
    #     train_std, test_std = p1.normalize(train, tuning_data)
    #     true_output_test = test_std[label]
    #     test_std = test_std.drop(columns=label)
    #
    #     input_weights, hidden_weights_1, hidden_weights_2 = p4.backprop_reg(train_std, label, hidden_nodes_1,
    #                                                                         hidden_nodes_2, learning_rate, epochs)
    #     pred = p4.predict_backprop_reg(test_std, hidden_nodes_1, hidden_nodes_2, input_weights, hidden_weights_1,
    #                                    hidden_weights_2)
    #
    #     accuracy = p1.evaluation_metrics(true_output=true_output_test, predicted_output=pred, metric_type=1)
    #     average_prediction.append(accuracy)
    #     print('Mean Squared Error on test: {:.4f} for eta: {}'.format(accuracy, learning_rate))
    #     print('*' * 50)
    #
    # print('Average Mean Squared Error across 5-folds: {:.4f} for eta: {} at {} epochs'.format(
    #     p1.avg_output(average_prediction), learning_rate, epochs))
    # # print('-' * 50)

    # Backpropagation Regression with Tuned Parameters

    hidden_nodes_1 = 5
    hidden_nodes_2 = 2
    learning_rate = 0.05
    epochs = 250

    folds = p1.k_fold_cross_validation(remaining_data, folds=5, stratified=False, class_label=label)
    average_prediction = []

    for i, x in enumerate(folds):
        test = x
        train_set = [l for idx, l in enumerate(folds) if idx != i]
        train = pd.DataFrame()
        for val in train_set:
            train = train.append(val)
        print('Running fold: {}'.format(i))
        print('Test size: {}, Train size: {}'.format(len(test), len(train)))

        train_std, test_std = p1.normalize(train, test)
        true_output_test = test_std[label]
        test_std = test_std.drop(columns=label)

        input_weights, hidden_weights_1, hidden_weights_2 = p4.backprop_reg(train_std, label, hidden_nodes_1,
                                                                            hidden_nodes_2, learning_rate, epochs)
        pred = p4.predict_backprop_reg(test_std, hidden_nodes_1, hidden_nodes_2, input_weights, hidden_weights_1,
                                       hidden_weights_2)

        accuracy = p1.evaluation_metrics(true_output=true_output_test, predicted_output=pred, metric_type=1)
        average_prediction.append(accuracy)
        print('Mean Squared Error on test: {:.4f} for eta: {}'.format(accuracy, learning_rate))
        print('*' * 50)

    print('Average Mean Squared Error across 5-folds: {:.4f} for eta: {} at {} epochs'.format(
        p1.avg_output(average_prediction), learning_rate, epochs))


