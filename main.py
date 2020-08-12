import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#D
#Loading the Breast Cancer dataset
X, y = datasets.load_breast_cancer(return_X_y=True)
X = np.array(X)
y = np.array(y)
X = X[50:200, [1, 3]] #rows 50 to 200, columns 1,3
y = y[50:200] #take places 50 to 200
y[y == 0] = -1 #replace every y=0 with y=-1
#Splitting the data to train and test sets, with test_size=0.3 and random_state=4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 4)

#E
#function that will plot SVM decision boundaries- from the tutorial#
def plot_svc_decision_function(model, ax=None, plot_support=True, color='black', label=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1])
    y = np.linspace(ylim[0], ylim[1])
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    CS = ax.contour(X, Y, P, colors=color,levels=[0], alpha=0.5,linestyles=['-'])
    if label:
        CS.collections[0].set_label(label)

    # plot support vectors
    if plot_support:
        # plot decision boundary and margins
        ax.contour(X, Y, P, colors=color,levels=[-1, 1], alpha=0.5,linestyles=['--', '--'])
        ax.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],
                                                                s=50, linewidth=1, facecolors='none', edgecolor=color)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

lambdas=[0.00001, 0.001, 0.1, 10, 10000]
m=len(y_train)
train_error=list() #for section F
test_error=list() #for section F
margin=list() #for section F
for lam in lambdas:
    model = SVC(kernel='linear', C=1/(2*m*lam))
    model.fit(X_train, y_train)

    fig,axs=plt.subplots(nrows=2, sharex=True, sharey=True)
    axs[0].set_title('SVM Results on the train data')
    axs[0].scatter(X_train[:, 0], X_train[:, 1], marker='*', c=y_train)
    axs[1].set_title('SVM results on test data')
    axs[1].scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test)
    plot_svc_decision_function(model, ax=axs[0], plot_support=True, color='red', label='SVM with $\lambda$ =' + str(lam))
    plot_svc_decision_function(model, ax=axs[1], plot_support=False, color='red', label='SVM with $\lambda$ =' + str(lam))
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')

    train_error.append(1-model.score(X_train,y_train)) #for section F- Return the mean accuracy on given X and y
    test_error.append(1-model.score(X_test,y_test)) #for section F- Return the mean accuracy on given X and y
    margin.append(1/np.linalg.norm(model.coef_[0])) #for section F

    plt.show()

#F
#for the train and test error
x=range(5)
fig,axs=plt.subplots(nrows=1,ncols=2, sharex=True, sharey=True)
plt.sca(axs[0])
plt.bar(np.array(x)-0.2,train_error,0.4, label='Train Error', color='green')
plt.bar(np.array(x)+0.2,test_error,0.4, label='Test Error', color='red')
plt.xticks(range(5), lambdas)
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Error')
plt.title('Train and Test Errors')
axs[0].legend(loc='upper left') #legend on the upper-left side of the graph
for i in range(len(train_error)):
    plt.text(x =x[i]-0.48, y =train_error[i]+0.01, s = np.around(train_error[i],decimals=3), size = 6)
    plt.text(x=x[i], y=test_error[i] + 0.01, s=np.around(test_error[i], decimals=3), size=6)

#for margin
plt.sca(axs[1])
plt.bar(np.array(x),margin,0.4)
plt.xticks(x, lambdas)
plt.yscale('log') #present in log scale
plt.xlabel('$\lambda$')
plt.ylabel('Margin Width (log)')
plt.title('The (One Sided) Margin Width')

for i in range(len(margin)):
    plt.text(x =x[i]-0.25, y =margin[i]+0.05, s = np.around(margin[i],decimals=3), size = 6)

plt.show()

"""we will choose lambda=0.1
It can be seen that when lambda=0.1, the margin is about the middle value and the test error is the smallest. That means,
we were able to do the best classification of the test set.
In the SVM algorithm the lambda is a regularization operator, and the smaller the lambda- the possibility of misclassification
and the bigger it is, the more we allow the algorithm to misclassify examples.
So in our model, we would prefer as few mistakes as possible in the training sample, but we still want the possibility 
to make the mistakes so we won't make too hard assumptions about our world.
In other words, it is possible to say that the lambda regularization operator allows us our learning space about the world.
When it is very small - the operator takes into account very small mistakes that could affect our decision unjustifiably,
and when the operator is too large - it does not consider our training set and ignores our world."""
