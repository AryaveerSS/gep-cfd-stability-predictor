import torch
def train_model(model, train_loader, val_loader, optimizer, scheduler, bce, mse, DEVICE, ALPHA, EPOCHS):
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0
        epoch_loss = 0.0
        for x, label, p in train_loader:
            x, label, p = x.to(DEVICE), label.to(DEVICE), p.to(DEVICE)

            optimizer.zero_grad()
            label_pred, p_pred = model(x)

            loss1 = bce(label_pred, label)
            loss2 = mse(p_pred, p)
            loss  = loss1 + ALPHA * loss2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = (torch.sigmoid(label_pred) > 0.5).float()
            correct     += (preds == label).sum().item()
            total       += label.size(0)
            epoch_loss  += loss.item()
        scheduler.step()
        train_acc = correct / total
        # train_acc_list.append(train_acc)

        # --- Validation ---
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for x, label, p in val_loader:
                x, label, p = x.to(DEVICE), label.to(DEVICE), p.to(DEVICE)  # BUG FIX: p on device
                label_pred, _ = model(x)
                preds   = (torch.sigmoid(label_pred) > 0.5).float()
                correct += (preds == label).sum().item()
                total   += label.size(0)

        val_acc = correct / total
        # val_acc_list.append(val_acc)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

        print(f"Epoch {epoch+1:02d} | Loss {epoch_loss/len(train_loader):.4f} "
          f"| Train {train_acc:.3f} | Val {val_acc:.3f} | LR {scheduler.get_last_lr()[0]:.2e}")


