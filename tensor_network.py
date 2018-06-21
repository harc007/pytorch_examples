import torch

N, D_in, H, D_out = 64, 500, 1000, 100

device = torch.device("cpu")
dtype = torch.float

X = torch.randn(N, D_in, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

epochs = 10
learning_rate = 1e-6

for i in range(epochs):
    h = X.mm(w1)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(w2)

    loss = (y-y_pred).pow(2).sum().item()
    print(i, loss)

    grad_y_pred = 2*(y-y_pred)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = X.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
