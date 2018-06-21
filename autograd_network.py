import torch

N, D_in, H, D_out = 64, 500, 1000, 100

device = torch.device("cpu")
dtype = torch.float

X = torch.randn(N, D_in, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
y = torch.randn(N, D_out, device=device, dtype=dtype)

epochs = 10
learning_rate = 1e-6

for i in range(epochs):
    y_pred = X.mm(w1).clamp(min=0).mm(w2)

    loss = (y-y_pred).pow(2).sum()
    print(i, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
