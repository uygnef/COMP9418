import scipy as sp
import numpy as np
import matplotlib.pyplot as pl
import sklearn as skl

def plot_image(array, dim=28):
    """
    Plot array as an image of dimensions dim * dim
    """
    img = array.reshape(dim,dim, order = "C")
    pl.imshow(img, cmap=pl.cm.gray)
    ax = pl.gca();ax.set_yticks([]);ax.set_xticks([])




def train_model(xtrain, ytrain, components):
    """
    Write your code here.
    """
    import sklearn.mixture
    import sklearn.decomposition

    #use PCA to preprocess data
    reducer = sklearn.decomposition.PCA(n_components=40)
    reducer.fit(xtrain)
    xtrain = reducer.transform(xtrain)



    model = []

    for i in range(10):
        sub_model = sklearn.mixture.GaussianMixture(n_components=components)
        sub_model.fit(xtrain[ytrain[:, i] == 1])
        model.append(sub_model)
        #  print(sub_model)

    model.append(reducer)
    # You can modify this to save other variables, etc
    # but make sure the name of the file is 'model.npz.
    np.savez_compressed(
        'model.npz',
        model=model,
    )


def make_predictions(xtest):
    """
    @param xtest: (Ntest,D)-array with test data
    @return class_pred: (N,C)-array with predicted classes using one-hot-encoding
    @return class_logprob: (N,C)-array with  predicted log probability of the classes
    """

    # Add your code here: You should load your trained model here
    # and write to the corresponding code for making predictions

    #load model
    model = np.load('model.npz');
    gmm_and_pca = model['model'].tolist();
    #use PCA
    render = gmm_and_pca[-1]
    model_list = gmm_and_pca[:-1]
    xtest = render.transform(xtest)


    class_logprob = []
    # calculate all probablities for 10 digit model
    for i, gmm in enumerate(model_list):
        class_logprob.append(np.ma.log(gmm.score_samples(xtest)))

    # select the largest probablity for each test data as predict value
    # class_pred = np.argmax(class_logprob, axis=0)
    class_logprob = np.array(class_logprob).T
    class_pred = np.zeros_like(class_logprob)
    class_pred[np.arange(len(class_logprob)), class_logprob.argmax(1)] = 1
    return class_pred, class_logprob

def predictive_performance(xdata, ydata, class_pred, class_logprob):
    """
    @param xdata:  (N,D)-array of features
    @param ydata:  (N,C)-array of one-hot-encoded true classes
    @class_pred: (N,C)-array of one-hot-encoded predicted classes
    @class_logprob: (N,C)-array of predicted class log probabilities
    """
    correct = np.zeros(xdata.shape[0])
    ltest = np.zeros(xdata.shape[0])
    for i, x in enumerate(xdata):
        correct[i] = np.all(ydata[i, :] == class_pred[i,:])
        ltest[i] = class_logprob[i, np.argmax(ydata[i,:])]
    accuracy = correct.mean()
    loglike = ltest.mean()
    return accuracy, loglike


data = np.load('mnist_train.npz')

# training data
xtrain = data['xtrain'][:50000]
ytrain = data['ytrain'][:50000]
xtest = data['xtrain'][50000:]
ytest = data['ytrain'][50000:]
import time

for i in range(1, 20):
    start = time.time()
    train_model(xtrain, ytrain, i)
    class_pred, class_logprob = make_predictions(xtest)
    accuracy, loglike = predictive_performance(xtest, ytest, class_pred, class_logprob)
    end = time.time()

    print(i, "componts: used ", end - start, " s")
    print ('Average test accuracy=' + str(accuracy))
    print ('Average test likelihood=' + str(loglike))
    print()
