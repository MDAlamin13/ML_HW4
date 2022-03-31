from matplotlib import pyplot as plt

par=[0,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
auc= [0.6392,    0.6297 ,   0.6995  ,  0.6794  ,  0.6450  ,  0.6230 ,   0.6220  ,  0.6220  ,  0.6220  ,  0.6220  ,  0.6220,    0.5000]
no_feature=[116,   106   , 14   ,  5     ,3    , 2   ,  1    , 1    , 1   ,  1     ,1 ,    0]

x_labels=['','.01','.1','.2','.3','.4','.5','.6','.7','.8','.9','1']
y_labels=[]
y_vals=[]
x=0.50
while(x<=0.76):
    y_labels.append(str(x))
    y_vals.append(x)
    x+=0.02
plt.xticks(par,x_labels)
#plt.yticks(y_vals,y_labels)
plt.plot(par,no_feature)
plt.xlabel('Value of regularization parameter (par)')
plt.ylabel('Number of Selected Features')
plt.title('Number of selected features for sparse logistic regression')
plt.savefig("q2_nuf.png")
