#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class perceptron:
    def __init__(self,learning_rate=0.1,n_iterations=500):
        self.lr=learning_rate
        self.epochs=n_iterations
        self.weights=None
        self.bias=None
        
    
    def fit(self,x,y):
        self.weights=np.zeros(x.shape[1])
        self.bias=0
        
        for epoch in range(self.epochs):
            for i in range(x.shape[0]):
                y_pred=self.activation(np.dot(self.weights,x[i])+self.bias)
                self.weights=self.weights+self.lr*(y[i]-y_pred)*x[i]
                self.bias=self.bias+self.lr*(y[i]-y_pred)
                
        print("training complete !")
        print(self.weights)
        print(self.bias)
        
    def activation(self,activation):
        if(activation>=0):
            return 1
        else:
            return 0
        
    def predict(self,x):
        y_pred=[]
        for i in range(x.shape[0]):
            y_pred.append(self.activation(np.dot(self.weights,x[0])+self.bias))
        return np.array(y_pred)
    
    
clf=perceptron()
x_np=np.array(x)
y_np=np.array(y)
clf.fit(x_np,y_np)

def plot_decision_regions(X, y, classifier):
    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    plt.xlabel('Math Score')
    plt.ylabel('Science Score')
    plt.title('Perceptron Decision Regions')
    plt.show()

# Assuming x_np and y_np are your feature matrix and labels respectively
plot_decision_regions(x_np, y_np, clf)


# In[ ]:




