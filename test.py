import torch

# Create a tensor
tensor = torch.zeros(4, 4)

# Indices where we want to insert values
indices = torch.tensor([[1, 0], [2, 2]])

# Values to be inserted
values = 11

# Scatter values at specified indices along dimension 0
tensor.scatter_(1, indices, values)

print(tensor)
t=torch.tensor([1,2])
for i in t:
    print(i)