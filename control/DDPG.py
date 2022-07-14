from NeuralNetwork import NN
import torch


a = NN.SimpleNN(5, 3, 1)
b = NN.SimpleNN(3, 3, 2)

input_s = torch.tensor([1, 2, 3], requires_grad=False, dtype=torch.float32)

print(input_s)
print(b(input_s))


q_sa_input = torch.cat((input_s, b(input_s)), 0)

print(a(q_sa_input))

optimizer = torch.optim.SGD(a.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(b.parameters(), lr=0.01)
optimizer.zero_grad()
optimizer2.zero_grad()
loss = -a(q_sa_input)
loss.backward()

optimizer.step()
optimizer2.step()

print("-----")
print(b(input_s))
print(a(q_sa_input))

x = 0
y = 0
z = 1

if (x == 1) | (y == 1) | (z == 1):
    print("asdf")

