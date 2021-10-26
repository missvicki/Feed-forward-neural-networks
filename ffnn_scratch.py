""" Victor Nomwesigwa, 2021 """
import numpy as np

from gen_data import gen_simple, gen_xor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from results import predictions, bias_layer_1, weights_layer2, weights_layer_1, bias_layer_2

def predict(X_test):
    #extract weights
    w1l1 = np.array(weights_layer_1)
    b1l1 = np.array(bias_layer_1)
    w2l2 = np.array(weights_layer2)
    b2l2 = np.array(bias_layer_2)
    
    output1 = X_test @ w1l1.T + b1l1.reshape(1,3)
    output1[output1<0] = 0
    output2 = output1 @ w2l2.T + b2l2.reshape(1,1)
    output = 1 / (1 + np.exp(-output2))
    return output

def main():
    """Run experiment."""
    X, Y = gen_xor(400)

    # Split into test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.25
    )

    output = predict(X_test)

    # collapse successes in the output, i.e. values that are >= 0.5 (round off to 1 and they are successes)
    accuracy_output = (output[output >= 0.5]).reshape(-1,1)
    
    # collapse successes in the prediction, i.e. values that are >= 0.5 (round off to 1 and they are successes)
    prediction = np.array(predictions)
    # sum all values in output that are collapsed to 1: these are the successes
    accuracy_prediction = (prediction[prediction >= 0.5]).reshape(-1,1)

    # cosine similarity to compare the values
    cosine_similarity = (accuracy_prediction / np.linalg.norm(accuracy_prediction)) * (
        accuracy_output / np.linalg.norm(accuracy_output)
    ).reshape(-1)
    cosine_similarity = np.sum(cosine_similarity)

    print(
        "The Cosine Similarity between the Numpy implementation and the PyTorch implementation is {:.2f}".format(cosine_similarity)
    )



if __name__ == "__main__":
    main()

