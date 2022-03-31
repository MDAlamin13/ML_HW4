from posixpath import splitdrive
import pandas as pd
import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt

stepsize=0.1

def split_data(X,n):
    x1=X[0:n]
    x2=X[n:]
    return [x1,x2]

def sigmoid(X):
    return expit(X)

## Calculate the gradient for +1/-1 encoding ##
def calculate_grad(data,label,w):
    N=data.shape[0]
    grad=data.T.dot(np.multiply(label,sigmoid(np.multiply(-label, data.dot(w))))) / float(-N)
    return grad

def logistic_train(X,Y,epsilon=0.00001,maxitr=1000):
    w=np.zeros(X.shape[1])
    pred=np.zeros(X.shape[0])
    itr=0
    while(itr<maxitr):
        grad=calculate_grad(X,Y,w)
        w_t=w-stepsize*grad
        pred_t=X.dot(w_t)
        diff=np.abs(pred_t-pred)
        avg_diff=np.mean(diff)
        w=w_t
        pred=pred_t
        if(avg_diff<epsilon):
            break
        itr+=1
    return w    
    

data = pd.read_csv('data.txt', sep="  ", header=None)
X=data.to_numpy()

col_1=np.ones(X.shape[0])
X=np.c_[X,col_1]

X=2*X-1     ### Converting the values for +1/-1 encoding ##


labels=pd.read_csv('labels.txt',sep=" ",header=None)
Y=labels[2].to_numpy()
Y=2*Y-1   ### Converting the values for +1/-1 encoding ##

X_TRAIN,X_TEST=split_data(X,2000)
Y_TRAIN,Y_TEST=split_data(Y,2000)

n_vals=[200,500,800,1000,1500,2000]
accuracies=[]

for n in n_vals:
    x_train,extra=split_data(X_TRAIN,n)
    y_train,extra=split_data(Y_TRAIN,n)
    
    w=logistic_train(x_train,y_train)
    predict=X_TEST.dot(w)
    predict[predict<=0]=-1
    predict[predict>0]=1
    diff=Y_TEST-predict
    miss=np.count_nonzero(diff)
    correct=Y_TEST.shape[0]-miss
    accuracy=(correct/Y_TEST.shape[0])*100
    accuracies.append(accuracy)

    
print(accuracies)
x_labels=['200','500','800','1000','1500','2000']
plt.xticks(n_vals,x_labels)
plt.plot(n_vals,accuracies)
plt.xlabel("Number of training data points")
plt.ylabel("Accuracy of the model(%)")
plt.title('Logistic regression model accuracy')
plt.savefig("q1.png")
