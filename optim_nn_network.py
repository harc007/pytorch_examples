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

learning_rate = 1e-4
epochs = 10

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for i in range(epochs):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(i, loss)

    optimizer.zero_grad()
    
    loss.backward()

    optimizer.step()
