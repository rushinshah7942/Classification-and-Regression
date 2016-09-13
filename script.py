import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    mu = []
    ytemp = y.reshape((y.shape[0],))
    for cl in np.unique(y):
        mu.append(np.mean(X[ytemp == cl], axis=0))
        
    means = np.array(mu)    
    covmat = np.dot((X-np.mean(means,0)).T,(X-np.mean(means,0)))/X.shape[0]
    
    return means.T,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    mu = []
    ytemp = y.reshape((y.shape[0],))
    for cl in np.unique(y):
        mu.append(np.mean(X[ytemp == cl], axis=0))
    
    means = np.array(mu)
    covmats = []
    for cl in np.unique(y):
        covmats.append(np.dot((X[ytemp==cl]-means[cl-1]).T,X[ytemp==cl]-means[cl-1])/X[ytemp==cl].shape[0])
        
    return means.T,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    sigma_inv = np.linalg.inv(covmat)
    ypred = np.zeros(ytest.shape)
    p_y = np.zeros((5,))
    for i in range(Xtest.shape[0]):
        for j in range(means.shape[1]):
            p_y[j] = np.dot(np.dot((Xtest[i]-means[:,j]).T,sigma_inv),(Xtest[i]-means[:,j]))
        ypred[i] = np.argmin(p_y) + 1
        
    acc = 100*np.mean((ypred==ytest).astype(float))

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    covmats_inv = []
    covmats_det = []
    D = Xtest.shape[1]
    for j in range(means.shape[1]):
        covmats_inv.append(np.linalg.inv(covmats[j]))
        covmats_det.append(np.linalg.det(covmats[j]))
    covmats_inv = np.array(covmats_inv)
    ypred = np.zeros(ytest.shape)
    p_y = np.zeros((5,))
    for i in range(Xtest.shape[0]):
        for j in range(means.shape[1]):
            p_y[j] = 1/((((2*pi)**D)*covmats_det[j])**0.5)*np.exp(-0.5*np.dot(np.dot((Xtest[i]-means[:,j]).T,covmats_inv[j]),(Xtest[i]-means[:,j])))
        ypred[i] = np.argmax(p_y) + 1
        
    acc = 100*np.mean((ypred==ytest).astype(float))
        
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD    

    w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))                                               
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    w = np.dot(np.linalg.inv(np.dot(X.T,X) + lambd*np.identity(X.shape[1])),np.dot(X.T,y))                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    rmse = sqrt(np.sum((ytest - np.dot(Xtest,w))**2)/ytest.shape[0])
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD

    
    w = w.reshape((w.flatten().shape[0],1))
    error = 0.5*(np.dot((y - np.dot(X,w)).T,(y - np.dot(X,w)))) + lambd*0.5*np.dot(w.T,w)
    error_grad = np.dot(np.dot(X.T,X),w) - np.dot(X.T,y) + lambd*w
    
    return error, error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD

    Xd = np.ones((x.shape[0],1))
    for i in range(1,p):
        Xd = np.concatenate((Xd,x.reshape((x.shape[0],1))**i),axis=1)
    
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,_ = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,_ = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2
plt.clf()
plt.cla()
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)        
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

#plt.cla()
#plt.clf()
#plt.plot(range(pmax),rmses5)
#plt.legend(('No Regularization','Regularization'))
