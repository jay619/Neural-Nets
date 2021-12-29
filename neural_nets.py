import pandas as pd
import numpy as np
import warnings
import data_pipeline as p1

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

breast_cancer_data = p1.load_data(file_path=p1.breast_cancer, has_column_headers=False,
                                  column_headers=p1.breast_cancer_headers, has_na=True, na_values=['?'])
# Dropping non-feature columns form the dataframe
breast_cancer_data = p1.drop_non_feature_columns(breast_cancer_data, column_labels=['sample_code_number'])
# Replacing NA values
breast_cancer_data = p1.replace_na_with_feature_mean(breast_cancer_data)

cars_data = p1.load_data(file_path=p1.cars, has_column_headers=False, column_headers=p1.car_headers, has_na=False)

house_votes84_data = p1.load_data(file_path=p1.house_votes84, has_column_headers=False,
                                  column_headers=p1.house_votes_headers, has_na=False)
house_votes_data = house_votes84_data[
    ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resoluton', 'physician_fee_freeze',
     'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban',
     'aid_to_nicaraguan_contras', 'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending',
     'superfund_right_to_sue', 'crime', 'duty_free_exports', 'export_administration_act_south_africa']]
house_votes_data = p1.categorical_encoding(house_votes_data, 1, None)
house_votes_class = house_votes84_data['class']
house_votes_data.insert(len(house_votes_data.columns), column='class', value=house_votes_class)

abalone_data = p1.load_data(file_path=p1.abalone, has_column_headers=False, column_headers=p1.abalone_headers,
                            has_na=False)
abalone_data = p1.categorical_encoding(dataframe=abalone_data, categorical_data_type=1, encoding_mapping=None)

comp_hardware_data = p1.load_data(file_path=p1.comp_hardware, has_column_headers=False,
                                  column_headers=p1.comp_hardware_headers, has_na=False)
# Dropping non-feature columns
comp_hardware_data = p1.drop_non_feature_columns(comp_hardware_data, column_labels=['vendor', 'model'])
comp_erp = comp_hardware_data['erp']  # Will be used later
comp_hardware_data = p1.drop_non_feature_columns(comp_hardware_data, column_labels=['erp'])

forest_fires_data = p1.load_data(file_path=p1.forest_fires, has_column_headers=True, has_na=False)
# Applying the log transformation to the area as suggested by the author
forest_fires_data['area'] = forest_fires_data['area'].apply(lambda a: np.log(a + 1))
# # Forest Fires categorical mapping
month_to_numerical = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                      'oct': 10, 'nov': 11, 'dec': 12}
day_to_numerical = {'sun': 0, 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6}
forest_fires_encoding = {'month': month_to_numerical, 'day': day_to_numerical}
# # Uncomment to apply ordinal encoding to the forest fires dataset based on the forest_fires_encoding
forest_fires_data = p1.categorical_encoding(dataframe=forest_fires_data, categorical_data_type=0,
                                            encoding_mapping=forest_fires_encoding)


# Only used for testing
def normalize(data):
    """
    Normalizes the data between 0 and 1
    @param data: Data that needs to be normalized
    @return: Returns the normalized data
    """
    max_val = data.max()
    min_val = data.min()
    return (data - min_val) / (max_val - min_val)


def predict_point(weights, unique_classes, test_data):
    """
    Used to make a prediction for a single data point (Logistic Regression - Classification)
    @param weights: Weights used to make the prediction
    @param unique_classes:
    @param test_data:
    @return: Returns the predicted class
    """
    # Predicting the output based on the weights
    o_i = []
    for i in range(len(unique_classes)):
        output = np.dot(weights[i].reshape(1, len(weights[0])), test_data.T)[0][0]
        o_i.append(output)
    # Applying soft-max to the output
    y_i = []
    for i in range(len(unique_classes)):
        val = np.exp(o_i[i]) / np.sum(np.exp(o_i))
        y_i.append(val)
    return unique_classes[np.argmax(y_i)]


def predict(weights, classes, test_data):
    """
    Make a prediction for every test point in the test data (Logistic Regression - Classification)
    @param weights: The learned weights used for making predictions
    @param classes: Unique classes in the data set
    @param test_data: Data for which classes needs to be predicted
    @return: Returns a prediction for every test point in the test data
    """
    pred = []
    for idx, row in test_data.iterrows():
        pred.append(predict_point(weights, classes, row.to_frame().T))
    return pred


def logistic_regression(training_data, class_headers, eta, max_iter, more_than_one_dummy=True):
    """
    Logistic Regression for classification problems
    @param training_data: Data that is used to learn the weights
    @param class_headers: Column header that stores the class predictions for each data point
    @param eta: Learning rate
    @param max_iter: Epochs to iterate over the training data
    @param more_than_one_dummy: Boolean: if the class header is one-hot encoded or not
    @return: The learned weights and the unique class values
    """
    # True Labels
    true_labels = training_data[class_headers]
    training_data = training_data.drop(columns=class_headers)
    # List of features
    features = len(training_data.columns)
    classes = None
    if more_than_one_dummy:
        classes = true_labels.columns
        unique_classes = len(true_labels.columns)
    else:
        classes = true_labels.unique()
        unique_classes = len(true_labels.unique())

    # Generating weights
    w_ij = np.random.uniform(-0.01, 0.01, size=(unique_classes, features))
    iterations = 0
    while iterations < max_iter:
        # To store weight changes over each epoch
        delta_w_ij = np.zeros(shape=(unique_classes, features))
        for idx, row in training_data.iterrows():
            # Calculating the output
            o_i = []
            for i in range(unique_classes):
                o_i.append(np.dot(w_ij[i], row.T))
            # Applying soft-max to the output
            y_i = []
            for i in range(unique_classes):
                y_i.append(np.exp(o_i[i]) / np.sum(np.exp(o_i)))
            # Computing the weight changes based on the prediction and the true class label
            for i in range(unique_classes):
                if more_than_one_dummy:
                    delta_w_ij[i] = delta_w_ij[i] + (true_labels.loc[idx][i] - y_i[i]) * row.T
                else:
                    delta_w_ij[i] = delta_w_ij[i] + (true_labels.loc[idx] - y_i[i]) * row.T
        # Updating the weights
        for i in range(unique_classes):
            w_ij[i] = w_ij[i] + eta * delta_w_ij[i]

        iterations += 1
    return w_ij, classes


def linear_regression(training_data, output_label, eta=0.05, max_iter=10):
    """
    Simple linear network for regression
    @param training_data: Data that is used to learn the weights
    @param output_label: Column header that stores the true output for each point in the training set
    @param eta: Learning rate
    @param max_iter: Epochs to iterate over the training data
    @return: Returns the learned weights
    """
    # Getting list of features
    true_output = training_data[output_label]
    # List of features
    features = training_data.columns[:-1]
    training_data = training_data.drop(columns=[output_label])
    # Randomizing data
    training_data = training_data.sample(frac=1)
    # To store weights
    w_i = []
    for i in range(len(features)):
        w_i.append(np.random.uniform(-0.01, 0.01))
    w_ij = np.array([w_i])

    iterations = 0
    while iterations < max_iter:
        # To store the weight changes over each epoch
        delta_w_ij = np.zeros(shape=(1, len(features)))

        for idx, row in training_data.iterrows():
            # Calculating the output
            o_i = np.dot(w_ij, row)[0]
            # Computing the weight changes
            delta_w_ij[0] = delta_w_ij[0] + (np.square((true_output[idx] - o_i)) * row.to_frame().T)
        # Updating the weights
        w_ij = w_ij + eta * np.divide(delta_w_ij, len(training_data.index))
        iterations += 1
    return w_ij[0]


def predict_output_simple_reg(test_data, weights):
    """
    Predicts the output for simple linear regression network based on the leanred weights
    @param test_data: Test data for which output needs to be predicted
    @param weights: Learned weights used to predict an output
    @return: Returns the predicted output for each data point in the test set
    """
    output = []
    for idx, row in test_data.iterrows():
        output.append(np.dot(weights, row.T))
    return output


def sigmoid(z):
    """
    Computes the Sigmoid function
    @param z: Value for which sigmoid needs to be calculated
    @return: Applies the sigmoid function to the input value (value between 0 and 1)
    """
    return 1 / (1 + np.exp(-z))


def tanh(z):
    """
    Computes the Hyperbolic Tangent function
    @param z: Value for which sigmoid needs to be calculated
    @return: Applies the tanh function to the input value (value between -1 and 1)
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def backprop(training_data, class_label, hidden_units_1, hidden_units_2, eta, max_iter):
    """
    Backpropagation to learn the weights in a neural network. Works only with two hidden layer network (Classification)
    @param training_data: Training data that is used to learn weights for each layer
    @param class_label: Column header that stores the class predictions for each data point
    @param hidden_units_1: Number of neurons in the first hidden layer
    @param hidden_units_2: Number of neurons in the second hidden layer
    @param eta: Learning rate
    @param max_iter: Epochs to iterate over the training data
    @return: Returns the learned weights across each layer and the unique classes
    """
    # Randomizing the training data
    training_data = training_data.sample(frac=1)
    # Storing the true class labels of the training dataset
    true_label = training_data[class_label]
    training_data = training_data.drop(columns=true_label.columns)
    # List of features
    features = training_data.columns
    # List of unique classes in the dataset
    unique_classes = true_label.columns
    # Generating weights
    # Input to Hidden 1
    w_hj = np.random.uniform(-0.01, 0.01, size=(hidden_units_1, len(features)))
    # Hidden 1 to Hidden 2
    v_h1h2 = np.random.uniform(-0.01, 0.01, size=(hidden_units_2, hidden_units_1))
    # Hidden 2 to output
    v_ih = np.random.uniform(-0.01, 0.01, size=(len(unique_classes), hidden_units_2))

    iterations = 0
    while iterations < max_iter:
        for idx, row in training_data.iterrows():
            # Hidden Layer 1
            z_h1 = []
            # Applying the tanh function to the output
            for h in range(hidden_units_1):
                z_h1.append(tanh(np.dot(w_hj[h], row.T)))

            o_i1 = []
            for i in range(len(unique_classes)):
                o_i1.append(np.dot(v_h1h2[i], z_h1))

            # Applying soft-max to the output
            y_i1 = []
            for i in range(len(unique_classes)):
                y_i1.append(np.exp(o_i1[i]) / np.sum(np.exp(o_i1)))

            # To store the weight changes in the second layer
            delta_vi1 = np.zeros(shape=(len(unique_classes), hidden_units_1))
            for i in range(len(unique_classes)):
                delta_vi1[i] = delta_vi1[i] + eta * (true_label.loc[idx][i] - y_i1[i]) * np.array(z_h1)

            # To store the weight changes in the first layer
            delta_wh1 = np.zeros(shape=(hidden_units_1, len(features)))
            for h in range(hidden_units_1):
                inner_sum1 = 0
                for i in range(len(unique_classes)):
                    inner_sum1 += (true_label.loc[idx][i] - y_i1[i]) * v_h1h2[i][h]
                delta_wh1[h] = eta * inner_sum1 * z_h1[h] * (1 - z_h1[h]) * row.T
            # Updating the weights for the second layer
            for i in range(len(unique_classes)):
                v_h1h2[i] = v_h1h2[i] + delta_vi1[i]
            # Updating the weights for the first layer
            for i in range(hidden_units_1):
                w_hj[i] = w_hj[i] + delta_wh1[i]

            # Hidden layer 2
            z_h2 = []
            # Applying the tanh function to the output
            for h in range(hidden_units_2):
                z_h2.append(tanh(np.dot(v_h1h2[h], z_h1)))

            o_i2 = []
            for i in range(len(unique_classes)):
                o_i2.append(np.dot(v_ih[i], z_h2))

            # Applying soft-max to the output
            y_i2 = []
            for i in range(len(unique_classes)):
                y_i2.append(np.exp(o_i2[i]) / np.sum(np.exp(o_i2)))

            # To store the weight changes in the third layer (output)
            delta_vi2 = np.zeros(shape=(len(unique_classes), hidden_units_2))
            for i in range(len(unique_classes)):
                delta_vi2[i] = delta_vi2[i] + eta * (true_label.loc[idx][i] - y_i2[i]) * np.array(z_h2)

            # To store the weight changes in the second layer
            delta_wh2 = np.zeros(shape=(hidden_units_2, hidden_units_1))
            for h in range(hidden_units_2):
                inner_sum2 = 0
                for i in range(len(unique_classes)):
                    inner_sum2 += (true_label.loc[idx][i] - y_i2[i]) * v_ih[i][h]
                delta_wh2[h] = eta * inner_sum2 * z_h2[h] * (1 - z_h2[h]) * np.array(z_h1)
            # Updating the weights for the third layer (output)
            for i in range(len(unique_classes)):
                v_ih[i] = v_ih[i] + delta_vi2[i]
            # Updating the weights for the second layer
            for i in range(hidden_units_2):
                v_h1h2[i] = v_h1h2[i] + delta_wh2[i]

        iterations += 1
    return w_hj, v_h1h2, v_ih, unique_classes


def predict_point_backprop(test_point, classes, hidden_neurons_1, hidden_neurons_2, input_weight, hidden_weight_1,
                           hidden_weight_2):
    """
    Predict a class for the test point (Backpropagation - Classification)
    @param test_point: Test point for which class needs to be predicted
    @param classes: Unique classes in the dataset
    @param hidden_neurons_1: Number of neurons in the first hidden layer
    @param hidden_neurons_2: Number of neurons in the second hidden layer
    @param input_weight: Learned weights from Input to first hidden layer
    @param hidden_weight_1: Learned weights from first hidden layer to the second hidden layer
    @param hidden_weight_2: Learned weights from the second hidden layer to the output layer
    @return: Returns the predicted class for the test point
    """
    # Compute the input for second layer
    z_h1 = []
    for h in range(hidden_neurons_1):
        z_h1.append(tanh(np.dot(input_weight[h], test_point)))

    o_i1 = []
    for i in range(len(classes)):
        o_i1.append(np.dot(hidden_weight_1[i], z_h1))

    # Apply soft-max to the output
    y_i1 = []
    for i in range(len(classes)):
        y_i1.append(np.exp(o_i1[i]) / np.sum(np.exp(o_i1)))

    # Compute the input for the thrid layer
    z_h2 = []
    for h in range(hidden_neurons_2):
        z_h2.append(tanh(np.dot(hidden_weight_1[h], z_h1)))

    o_i2 = []
    for i in range(len(classes)):
        o_i2.append(np.dot(hidden_weight_2[i], z_h2))

    # Apply soft-max to the output
    y_i2 = []
    for i in range(len(classes)):
        y_i2.append(np.exp(o_i2[i]) / np.sum(np.exp(o_i2)))

    return classes[np.argmax(y_i2)]


def predict_backprop(test_data, classes, hidden_neurons_1, hidden_neurons_2, input_weight, hidden_weight_1,
                     hidden_weight_2):
    """
    Predict a class for the every test point in the test data set (Backpropagation - Classification)
    @param test_data: Test data for which classes need to be predicted
    @param classes: Unique classes in the dataset
    @param hidden_neurons_1: Number of neurons in the first hidden layer
    @param hidden_neurons_2: Number of neurons in the second hidden layer
    @param input_weight: Learned weights from Input to first hidden layer
    @param hidden_weight_1: Learned weights from first hidden layer to the second hidden layer
    @param hidden_weight_2: Learned weights from the second hidden layer to the output layer
    @return: Returns the predicted class for each point in the test data set
    """
    pred = []
    for idx, row in test_data.iterrows():
        pred.append(
            predict_point_backprop(row.T, classes, hidden_neurons_1, hidden_neurons_2, input_weight, hidden_weight_1,
                                   hidden_weight_2))
    return pred


def backprop_reg(training_data, output_label, hidden_units_1, hidden_units_2, eta, max_iter):
    """
    Backpropagation to learn the weights in a neural network. Works only with two hidden layer network (Regression)
    @param training_data: Training data used to learn the weights at each layer
    @param output_label: Column header that stores the true output
    @param hidden_units_1: Number of neurons in the first hidden layer
    @param hidden_units_2: Number of neurons in the second hidden layer
    @param eta: Learning rate
    @param max_iter: Epochs to iterate over the training data
    @return: Returns the learned weights across each layer and the unique classes
    """
    # Randomizing the training data
    training_data = training_data.sample(frac=1)
    # Storing the true outputs
    true_output = training_data[output_label]
    training_data = training_data.drop(columns=output_label)
    # List of features
    features = training_data.columns

    # Generating weights
    # Input to Hidden 1
    w_hj = np.random.uniform(-0.01, 0.01, size=(hidden_units_1, len(features)))
    # Hidden 1 to Hidden 2
    v_h1h2 = np.random.uniform(-0.01, 0.01, size=(hidden_units_2, hidden_units_1))
    # Hidden 2 to output
    v_ih = np.random.uniform(-0.01, 0.01, size=(1, hidden_units_2))

    iterations = 0
    while iterations < max_iter:
        for idx, row in training_data.iterrows():
            # Hidden Layer 1
            z_h1 = []
            # Applying the tanh function
            for h in range(hidden_units_1):
                z_h1.append(tanh(np.dot(row.T, w_hj[h])))

            # Computing the output
            y_i1 = []
            for k in range(hidden_units_2):
                y_i1.append(np.dot(v_h1h2[k].T, z_h1))

            # To store the weight changes in the second layer
            delta_vi1 = np.zeros(shape=(hidden_units_2, hidden_units_1))
            for i in range(hidden_units_2):
                delta_vi1[i] = delta_vi1[i] + eta * (true_output.loc[idx] - y_i1[i]) * np.array(z_h1)

            # To store the weight changes in the first layer
            delta_wh1 = np.zeros(shape=(hidden_units_1, len(features)))
            for h in range(hidden_units_1):
                inner_sum1 = 0
                for i in range(hidden_units_2):
                    inner_sum1 += np.sum((true_output.loc[idx] - y_i1[i]) * v_h1h2[i][h])
                delta_wh1[h] = eta * inner_sum1 * z_h1[h] * (1 - z_h1[h]) * row.T

            # Updating the weights for the second layer
            for i in range(hidden_units_2):
                v_h1h2[i] = v_h1h2[i] + delta_vi1[i]

            # Updating the weights for the first layer
            for h in range(hidden_units_1):
                w_hj[h] = w_hj[h] + delta_wh1[h]

            # Hidden Layer 2
            z_h2 = []
            # Applying the tanh function
            for h in range(hidden_units_2):
                z_h2.append(tanh(np.dot(v_h1h2[h], z_h1)))

            # Computing the output
            y_i2 = np.dot(v_ih, z_h2)

            # Weight changes in the third layer
            delta_vi2 = eta * (true_output.loc[idx] - y_i2) * z_h2

            # To store the weight changes in the second layer
            delta_wh2 = np.zeros(shape=(hidden_units_2, hidden_units_1))
            for h in range(hidden_units_2):
                delta_wh2[h] = eta * np.sum((true_output.loc[idx] - y_i2) * v_ih) * z_h2[h] * (1 - z_h2[h]) * np.array(
                    z_h1)

            # Updating the weights to the output layer
            v_ih = v_ih + delta_vi2

            # Updating the weights for the second layer
            for h in range(hidden_units_2):
                v_h1h2[h] = v_h1h2[h] + delta_wh2[h]

        iterations += 1
    return w_hj, v_h1h2, v_ih


def predict_point_backprop_reg(test_point, hidden_neurons_1, hidden_neurons_2, input_weight, hidden_weight_1,
                               hidden_weight_2):
    """
    Predict an output for the test point (Backpropagation - Regression)
    @param test_point: Test point for which output needs to be predicted
    @param hidden_neurons_1: Number of hidden neurons in the first hidden layer
    @param hidden_neurons_2: Number of hidden neurons in the second hidden layer
    @param input_weight: Learned weights from Input to first hidden layer
    @param hidden_weight_1: Learned weights from first hidden layer to the second hidden layer
    @param hidden_weight_2: Learned weights from the second hidden layer to the output layer
    @return: Returns the predicted output for the test point
    """
    # Compute the input to the second layer
    z_h1 = []
    for h in range(hidden_neurons_1):
        z_h1.append(tanh(np.dot(input_weight[h], test_point)))

    # Compute the output
    y_i1 = []
    for k in range(hidden_neurons_2):
        y_i1.append(np.dot(hidden_weight_1[k].T, z_h1))

    # Compute the input to the third layer
    z_h2 = []
    for h in range(hidden_neurons_2):
        z_h2.append(tanh(np.dot(hidden_weight_1[h], z_h1)))

    # Compute the output
    y_i = np.dot(hidden_weight_2, z_h2)
    return y_i[0]


def predict_backprop_reg(test_data, hidden_neurons_1, hidden_neurons_2, input_weight, hidden_weight_1, hidden_weight_2):
    """
    Predict an output for the every test point in the test data set (Backpropagation - Regression)
    @param test_data: Test data for which outputs need to be predicted
    @param hidden_neurons_1: Number of neurons in the first hidden layer
    @param hidden_neurons_2: Number of neurons in the second hidden layer
    @param input_weight: Learned weights from Input to first hidden layer
    @param hidden_weight_1: Learned weights from first hidden layer to the second hidden layer
    @param hidden_weight_2: Learned weights from the second hidden layer to the output layer
    @return: Returns the predicted output for each point in the test data set
    """
    pred = []
    for idx, row in test_data.iterrows():
        pred.append(predict_point_backprop_reg(row.T, hidden_neurons_1, hidden_neurons_2, input_weight, hidden_weight_1,
                                               hidden_weight_2))
    return pred


def get_compressed_data(training_data, encoding_weight):
    """
    Compresses the data from the original input space to the updated input space using the encoding weights
    @param training_data: The original training data
    @param encoding_weight: Weights trained by the autoencoder
    @return: Returns the new dataset in the new input space
    """
    encoded_data = []
    for idx, row in training_data.iterrows():
        encoded_data.append(tanh(np.dot(encoding_weight, row)))
    return encoded_data


def encoder(training_data, hidden_units, eta, max_iter):
    """
    Autoencoder based on the number of hidden neurons required for the hidden layer
    @param training_data: The Training data that needs to be compressed to a smaller input space
    @param hidden_units: Number of neurons needed for the hidden layer (should be less than the original input features)
    @param eta: learning rate
    @param max_iter: total epochs to iterate over the training data
    @return: Returns the learned weights for the encoding layer (input to hidden layer)
    """
    features = training_data.columns
    # Input to Hidden 1
    w_hj = np.random.uniform(-0.01, 0.01, size=(hidden_units, len(features)))
    # Hidden 2 to output (features)
    v_ih = np.random.uniform(-0.01, 0.01, size=(len(features), hidden_units))

    iterations = 0
    while iterations < max_iter:
        for idx, row in training_data.iterrows():
            z_h = []
            # Applying the tanh function
            for h in range(hidden_units):
                z_h.append(tanh(np.dot(w_hj[h], row.T)))
            # Computing the outputs
            o_i = []
            for i in range(len(features)):
                o_i.append(np.dot(v_ih[i], z_h))

            # To store weight changes in the second layer (hidden to output)
            delta_vi = np.zeros(shape=(len(features), hidden_units))
            for i in range(len(features)):
                delta_vi[i] = delta_vi[i] + eta * (row[i] - o_i[i]) * np.array([z_h])

            # To store weight changes in the first layer (input to hidden)
            delta_wh = np.zeros(shape=(hidden_units, len(features)))
            for h in range(hidden_units):
                inner_sum = 0
                for i in range(len(features)):
                    inner_sum += (row[i] - o_i[i]) * v_ih[i][h]
                delta_wh[h] = delta_wh[h] + eta * inner_sum * z_h[h] * (1 - z_h[h]) * row.T

            # Updating the weights for the second layer
            for i in range(len(features)):
                v_ih[i] = v_ih[i] + delta_vi[i]

            # Updating the weights for the first layer
            for h in range(hidden_units):
                w_hj[h] = w_hj[h] + delta_wh[h]
        iterations += 1
    return w_hj


def autoencoder_class(training_data, class_label, hidden_units_1, hidden_units_2, eta, max_iter):
    """
    Applies auto-encoding to the first hidden layer. The second hidden layer is just a
    regular hidden layer trained via backpropagation (Classification)
    @param training_data: Training data used to learn the weights for each layer
    @param class_label: Column header that stores the class predictions for each data point
    @param hidden_units_1: Number of neurons in the first hidden layer
    @param hidden_units_2: Number of neurons in the second hidden layer
    @param eta: learning rate
    @param max_iter: Epochs to iterate over the training data
    @return: Returns the learned weights at each layer. The weights at the first layer are learned via autoencoder
    """
    # Storing the true class labels
    true_label = training_data[class_label]
    training_data = training_data.drop(columns=true_label.columns)
    # List of features
    features = training_data.columns
    # List of unique classes in the dataset
    unique_classes = true_label.columns
    # Weights for the first layer learned via the autoencoder
    encoded_weights = encoder(training_data, hidden_units_1, eta, max_iter)
    # New dataset which with reduced input features based on weights learned by the autoencoder
    encoded_dataset = get_compressed_data(training_data, encoded_weights)
    encoded_dataset = pd.DataFrame(encoded_dataset, index=true_label.index)
    # Input to Hidden
    w_hj = np.random.uniform(-0.01, 0.01, size=(hidden_units_2, len(encoded_dataset.columns)))
    # Hidden to output
    v_ih = np.random.uniform(-0.01, 0.01, size=(len(unique_classes), hidden_units_2))

    iterations = 0
    while iterations < max_iter:
        for idx, row in encoded_dataset.iterrows():
            # Applying the tanh function to the input for the output layer
            z_h = []
            for h in range(hidden_units_2):
                z_h.append(tanh(np.dot(w_hj[h], row.T)))

            # Computing the output
            o_i = []
            for i in range(len(unique_classes)):
                o_i.append(np.dot(v_ih[i], z_h))

            # Applying soft-max to the output
            y_i = []
            for i in range(len(unique_classes)):
                y_i.append(np.exp(o_i[i]) / np.sum(np.exp(o_i)))

            # To store weight changes from the second hidden layer to the output
            delta_vi = np.zeros(shape=(len(unique_classes), hidden_units_2))
            for i in range(len(unique_classes)):
                delta_vi[i] = delta_vi[i] + eta * (true_label.loc[idx][i] - y_i[i]) * np.array(z_h)

            # To store the weight changes from the first hidden layer to the second hidden layer
            delta_wh = np.zeros(shape=(hidden_units_2, len(encoded_dataset.columns)))
            for h in range(hidden_units_2):
                inner_sum = 0
                for i in range(len(unique_classes)):
                    inner_sum += (true_label.loc[idx][i] - y_i[i]) * v_ih[i][h]
                delta_wh[h] = eta * inner_sum * z_h[h] * (1 - z_h[h]) * row.T

            # Updating the weights for the third layer (Hidden to output)
            for i in range(len(unique_classes)):
                v_ih[i] = v_ih[i] + delta_vi[i]
            # Updating the weights for the second layer (Hidden 1 to hidden 2)
            for i in range(hidden_units_2):
                w_hj[i] = w_hj[i] + delta_wh[i]

        iterations += 1
    return encoded_weights, w_hj, v_ih, unique_classes


def autoencoder_reg(training_data, class_label, hidden_units_1, hidden_units_2, eta, max_iter):
    """
    Applies auto-encoding to the first hidden layer. The second hidden layer is just a
    regular hidden layer trained via backpropagation (Regression)
    @param training_data: Training data used to learn the weights for each layer
    @param class_label: Column header that stores the true output for each data point
    @param hidden_units_1: Number of neurons in the first hidden layer
    @param hidden_units_2: Number of neurons in the second hidden layer
    @param eta: learning rate
    @param max_iter: Epochs to iterate over the training data
    @return: Returns the learned weights at each layer. The weights at the first layer are learned via autoencoder
    """
    # Getting list of features
    true_output = training_data[class_label]
    # List of features
    training_data = training_data.drop(columns=class_label)
    features = training_data.columns

    # Weights for the first layer learned via the autoencoder
    encoded_weights = encoder(training_data, hidden_units_1, eta, max_iter)
    # New dataset which with reduced input features based on weights learned by the autoencoder
    encoded_dataset = get_compressed_data(training_data, encoded_weights)
    encoded_dataset = pd.DataFrame(encoded_dataset, index=true_output.index)
    # Input to Hidden
    w_hj = np.random.uniform(-0.01, 0.01, size=(hidden_units_2, hidden_units_1))
    # Hidden to output
    v_ih = np.random.uniform(-0.01, 0.01, size=(1, hidden_units_2))

    iteration = 0
    while iteration < max_iter:
        for idx, row in encoded_dataset.iterrows():
            # Applying the tanh function to the input for the output layer
            z_h = []
            for h in range(hidden_units_2):
                z_h.append(tanh(np.dot(w_hj[h], row.T)))

            # Computing the output
            y_i = np.dot(v_ih, z_h)

            # To store weight changes from the second hidden layer to the output
            delta_vi = eta * (true_output.loc[idx] - y_i) * z_h

            # To store the weight changes from the first hidden layer to the second hidden layer
            delta_wh = np.zeros(shape=(hidden_units_2, hidden_units_1))
            for h in range(hidden_units_2):
                delta_wh[h] = eta * np.sum((true_output.loc[idx] - y_i) * v_ih) * z_h[h] * (1 - z_h[h]) * row.T

            # Updating the weights for the third layer (Hidden to output)
            v_ih = v_ih + delta_vi
            # Updating the weights for the second layer (Hidden 1 to hidden 2)
            for h in range(hidden_units_2):
                w_hj[h] = w_hj[h] + delta_wh[h]

        iteration += 1
    return encoded_weights, w_hj, v_ih


if __name__ == '__main__':
    print('Please run the individual test files for running the experiments with different models ...')
