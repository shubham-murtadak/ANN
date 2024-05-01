#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class McCullochPittsNeuron:
    
    def __init__(self,threshold,weights):
        self.threshold=threshold
        self.weights=weights
        
        
    
    def activate(self,inputs):
        weighted_sum=sum(w * x for w,x in zip(self.weights,inputs))
        
        if(weighted_sum>=self.threshold):
            return 1
        
        else:
            return 0
        
        
        
def and_not(x1,x2):
    neuron=McCullochPittsNeuron(threshold=1,weights=[1,-1])
    
    active=neuron.activate([x1,x2])

    print(f"\nOutput of ANDNOT GATE :({x1},{x2})=",active )
    
    
if __name__ == "__main__":
    print("Welcome to ANDNOT GATE USING McCullochPitts Model :")
    
    x1 = int(input('Enter value of x1: '))
    x2 = int(input('Enter value of x2: '))
    
    and_not(x1, x2)
    
    
    


# In[ ]:




