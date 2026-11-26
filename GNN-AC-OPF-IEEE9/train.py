# Lead project author and repository maintainer: Muhammad Abdullah Anwar
"""Train and evaluate the IEEE 9-bus GNN surrogate."""
import argparse
import json
from pathlib import Path
import torch
from dataset import get_ieee9_loaders, in_channels, out_channels
from model import IEEE9_GNN, MSE


def evaluate(model, loader, stats, device):
    model.eval()
    mean = stats['y_mean'].to(device).view(1, 4)
    std = stats['y_std'].to(device).view(1, 4)
    sums = torch.zeros(4, device=device)
    counts = torch.zeros(4, device=device)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch) * std + mean
            true = batch.y * std + mean
            error = (pred - true).square()
            gen = torch.arange(batch.num_nodes, device=device) % 9 < 3
            sums[:2] += error[:, :2].sum(dim=0)
            sums[2:] += error[gen, 2:].sum(dim=0)
            counts[:2] += batch.num_nodes
            counts[2:] += gen.sum()
    return dict(zip(['voltage_pu', 'angle_deg', 'active_power_mw', 'reactive_power_mvar'], (sums / counts).sqrt().cpu().tolist()))


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- Get loaders from Excel ---
    train_loader, val_loader, stats = get_ieee9_loaders(excel_path=args.data, batch_size=args.batch_size)

    # --- Initialize model & optimizer ---
    model = IEEE9_GNN(in_channels=in_channels, out_channels=out_channels).to(device)

    lambda_l2 = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=lambda_l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20
    )

    epochs = args.epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            y_pred = model(batch)   # [num_nodes_total, 4]
            target = batch.y        # [num_nodes_total, 4] normalized

            loss = MSE(y_pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_pred = model(batch)
                target = batch.y
                val_loss += MSE(y_pred, target).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

    metrics = evaluate(model, val_loader, stats, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'stats': stats}, output / 'model.pt')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/IEEE9_OPF_Dataset_1.xlsx')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    main(args)
