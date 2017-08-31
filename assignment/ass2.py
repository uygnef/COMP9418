# Necessary libraries
import scipy as sp
import numpy as np
import matplotlib.pyplot as pl
import sklearn as skl

# Put the graphs where we can see them
def plot_image(array, dim=28):
    """
    Plot array as an image of dimensions dim * dim
    """
    img = array.reshape(dim,dim, order = "C")
    pl.imshow(img, cmap=pl.cm.gray)
    ax = pl.gca();ax.set_yticks([]);ax.set_xticks([])

data = np.load('mnist_train.npz')

# training data
xtrain = data['xtrain']
ytrain = data['ytrain']

idx = 10
plot_image(xtrain[idx,:])
print(ytrain[idx,:])

def train_model(xtrain, ytrain):
    """
    Write your code here.
    """
    from sklearn import mixture
    gmm = mixture.GaussianMixture(n_components=10)
    gmm.fit(xtrain, y = ytrain)
    # You can modify this to save other variables, etc
    # but make sure the name of the file is 'model.npz.
    np.savez_compressed(
        'model.npz',
        model=gmm,
    )

train_model(xtrain, ytrain)