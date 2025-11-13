# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:29:29 2025

@author: gauthambekal93
"""

import torch 
'''
x = torch.tensor([5.0], requires_grad=True)
y = x.clone()  # y still part of graph
z = y * 2
z.backward()
print(x.grad)  # ✅ works
'''

'''
x = torch.tensor([5.0], requires_grad=True)
y = x.clone().requires_grad_(True)
z = y * 3
z.backward()
print("X gradient is: ",x.grad)  # ❌ None
print("Y gradient is: ",y.grad)  # ✅ tensor([3.])
'''

'''
x = torch.tensor([2.0], requires_grad=True)
y = x.clone()
z = y * 3
z.backward()
print("X gradient is: ",x.grad)  
print("Y gradient is: ",y.grad)  
'''

'''
x = torch.tensor([2.0], requires_grad=True)
y = x.clone().requires_grad_(True)
z = y * 3
z.backward()

print("X gradient is: " ,x.grad)  # ❌ None (because y is a leaf now; graph starts from y)
print("Y gradient is: ", y.grad)  # ✅ tensor([3.])
'''


'''
x = torch.tensor([2.0], requires_grad=True)
y = x * 2
z = y.clone().requires_grad_(True)  # New leaf tensor
out = z * 3
out.backward()

print("X gradient is:", x.grad)  # ❌ None
print("Z gradient is:",z.grad)  # ✅ Works
'''


'''
import torch
b = torch.tensor([1.0])
a = b.clone()           # No new tensor created

a += 2
print(b)        # ✅ b is also changed
'''


'''
x = torch.tensor([2.0], requires_grad=True)
y = x.clone().requires_grad_(True)

z = y * 3
z.backward()
print("x grad output: ",x.grad) 
print("y grad output: ",y.grad)  

'''


import torch

# Define a simple computation
x = torch.tensor([2.0], requires_grad=True)
y = x.clone().requires_grad_(True)
z = y ** 2
loss = z.mean()

# Function to trace backward graph
def trace_graph(tensor):
    print("Tensor: ", tensor)
    seen = set()
    def _trace(fn):
        if fn is None or fn in seen:
            return
        seen.add(fn)
        print(fn)
        for next_fn, _ in fn.next_functions:
            _trace(next_fn)
    _trace(tensor.grad_fn)

# Trace the graph starting from the loss
trace_graph(loss)
