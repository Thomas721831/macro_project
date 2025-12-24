import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import load_macro_data
from models import LinearRegressor, MLPRegressor
from utils import compute_loss, evaluate

from utils import predict_with_uncertainty

mean, std = predict_with_uncertainty(mlp_model, test_loader, device, n_samples=50)
print("Example test prediction + uncertainty:")
for i in range(5):
    print(f"y_pred = {mean[i].item():.3f} ± {2*std[i].item():.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_cols = ['gdp_growth', 'unemployment', 'interest_rate', '...']
    target_col = 'inflation'

    train_ds, val_ds, test_ds, df = load_macro_data(
        "../data/macro_data.csv",
        feature_cols,
        target_col
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    input_dim = len(feature_cols)

    # ---------- 1) Linear regression (no reg) ----------
    lin_model = LinearRegressor(input_dim).to(device)
    optimizer = optim.Adam(lin_model.parameters(), lr=1e-2)

    for epoch in range(50):
        lin_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = lin_model(x)
            loss = compute_loss(pred, y)  # plain MSE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate(lin_model, val_loader, device)
        if (epoch + 1) % 10 == 0:
            print(f"[Linear] Epoch {epoch+1}, val MSE: {val_loss:.4f}")

    test_loss = evaluate(lin_model, test_loader, device)
    print(f"[Linear] Test MSE: {test_loss:.4f}")

    # ---------- 2) Linear regression + L2 (Ridge) ----------
    ridge_model = LinearRegressor(input_dim).to(device)
    optimizer = optim.Adam(ridge_model.parameters(), lr=1e-2)

    l2_lambda = 1e-4

    for epoch in range(50):
        ridge_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = ridge_model(x)
            loss = compute_loss(pred, y, model=ridge_model, l2_lambda=l2_lambda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate(ridge_model, val_loader, device)
        if (epoch + 1) % 10 == 0:
            print(f"[Ridge] Epoch {epoch+1}, val MSE: {val_loss:.4f}")

    test_loss = evaluate(ridge_model, test_loader, device)
    print(f"[Ridge] Test MSE: {test_loss:.4f}")

    # ---------- 3) MLP regressor ----------
    mlp_model = MLPRegressor(input_dim, hidden_dims=[64, 64], dropout=0.1).to(device)
    optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)

    for epoch in range(100):
        mlp_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = mlp_model(x)
            loss = compute_loss(pred, y, model=mlp_model, l2_lambda=1e-5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate(mlp_model, val_loader, device)
        if (epoch + 1) % 10 == 0:
            print(f"[MLP] Epoch {epoch+1}, val MSE: {val_loss:.4f}")

    test_loss = evaluate(mlp_model, test_loader, device)
    print(f"[MLP] Test MSE: {test_loss:.4f}")


if __name__ == "__main__":
    main()