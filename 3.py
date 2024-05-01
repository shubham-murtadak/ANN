#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class perceptron:
    def __init__(self,learning_rate=0.1,n_iterations=200):
        self.lr=learning_rate
        self.epochs=n_iterations
        self.weights=None
        self.bias=None
        
    def fit(self,x,y):
        self.weights=np.zeros(x.shape[1])
        self.bias=0
        
        for i in range(self.epochs):
            converge=True
            for i in range(x.shape[0]):
                y_pred=self.activation(np.dot(self.weights,x[i])+self.bias)
                
                if y_pred!=y[i]:
                    self.weights=self.weights+self.lr*(y[i]-y_pred)*x[i]
                    self.bias=self.bias+self.lr*(y[i]-y_pred)
                    print(self.weights,self.bias)
                    converge=False
                    
            if converge:
                print("converge after epoch :",epoch)
                break
                
    def predict(self,x):
        y_pred=[]
        
        for i in range(x.shape[0]):
            y_pred.append(self.activation(np.dot(self.weights,x[i])+self.bias))
            
        return y_pred
    
    def activation(self,activ):
        if(activ>=0):
            return 1
        else:
            return 0
        
            
def ascii_value(a,n):
    ascii_rep=[]
    for i in range(a,n):
        binray_value=ord(str(i))
#         print(binray_value)
        ascii_val=[int(bit) for bit in format(binray_value,'08b')]
        ascii_rep.append(ascii_val)
        
    return ascii_rep


        
def label(a,n):
    label_rep=[]
    for i in range(a,n):
        if(i%2==0):
            label_rep.append(0)
        else:
            label_rep.append(1)
            
    return label_rep
        
        
input_x=np.array(ascii_value(0,10))
input_y=np.array(label(0,10))
# print(input_data)
test_data=np.array(ascii_value(3,6))
test_label=np.array(label(3,6))
print(test_data)
print(test_label) 
clf=perceptron()
clf.fit(input_x,input_y)
clf.predict(test_data)




# In[ ]:




