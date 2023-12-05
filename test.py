import torch
import numpy as np
# Create a tensor
tensor = torch.zeros(4, 4)

# Indices where we want to insert values
indices = torch.tensor([[1,2, 3], [4,5,6],[7,8,9]])
y=indices[torch.arange(3),torch.Tensor([1,2,1]).to(torch.int32)]
print(y)
indices = torch.tensor([1,2, 3])
mask=indices>2
print(mask)
print(indices[mask])
indices = torch.tensor([[1,2, 3], [4,5,6],[7,8,9]])
indices=np.asarray(indices)
key=lambda x: x.data.tobytes()
for index,label in enumerate(indices):
    print(index)
    print(label)
    
    print(key(label))
print(indices.data.tobytes())
a=[[113, 113, 113, 113, 113, 113, 113, 113, 113, 113],
        [113, 113, 113, 113, 113, 113, 113, 113, 113,  48]]
a=np.array(a)
print(a[0].data.tobytes()==a[0].data.tobytes())
# print(a[1].data.tobytes())
# # Values to be inserted
# values = 11

# # Scatter values at specified indices along dimension 0
# tensor.scatter_(1, indices, values)

# print(tensor)
# t=torch.tensor([1,2])
# for i in t:
#     print(i)

from sklearn.neighbors import KDTree
import numpy as np

X = np.array([[-2, -2], [-1, -1], [-2, -1], [-3, -2], [0, 0],
              [1, 1], [2, 1], [2, 2], [3, 2]])
kdt = KDTree(X, leaf_size=3, metric='euclidean')
tree_data, index, tree_nodes, node_bounds = kdt.get_arrays()
print(kdt.get_arrays()[1])
a=[1,2,3]
b=[4,5,6]
# m,n=zip(a,b)
# print(n)
a=[1,2,3]
print(torch.tensor(a).sum())

c=np.unique([1,1,2,3])
print(len(c))
tensor = torch.tensor([[1, 1, 3, 4],
                       [3, 4, 5, 6],
                       [1, 2, 3, 4]])

# Find unique elements in each row
unique_elements_per_row = torch.unique(tensor)

print(unique_elements_per_row)
# Possibility theory uses the possibility distribution to model uncertainty problems. Possibility distribution assign possibilities to different events, corresponding to a mapping function from single event to the scale [0,1]. It solves the equation:​
# ​
# ​