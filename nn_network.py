import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out)
        )

loss_fn = torch.nn.MSELoss(size_average=False)

epochs = 10
learning_rate = 1e-4

for i in range(epochs):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(i, loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate*param.grad
