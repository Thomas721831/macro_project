import torch

def compute_loss(pred, target, model=None, l2_lambda=0.0, l1_lambda=0.0):
    mse = torch.mean((pred - target) ** 2)

    l2_reg = torch.tensor(0.0, device=pred.device)
    l1_reg = torch.tensor(0.0, device=pred.device)

    if model is not None:
        for p in model.parameters():
            if p.requires_grad:
                if l2_lambda > 0:
                    l2_reg = l2_reg + torch.sum(p ** 2)
                if l1_lambda > 0:
                    l1_reg = l1_reg + torch.sum(torch.abs(p))

    loss = mse + l2_lambda * l2_reg + l1_lambda * l1_reg
    return loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size
    return total_loss / n

def predict_with_uncertainty(model, dataloader, device, n_samples=50):
    model.train()  # enable dropout!
    preds_all = []

    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for x, _ in dataloader:
                x = x.to(device)
                pred = model(x)
                batch_preds.append(pred.cpu())
            preds_all.append(torch.cat(batch_preds, dim=0))

    preds_all = torch.stack(preds_all, dim=0)  # [n_samples, N, 1]
    mean = preds_all.mean(dim=0)
    std = preds_all.std(dim=0)
    return mean, std