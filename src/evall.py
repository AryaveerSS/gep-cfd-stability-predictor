import torch
from sklearn.metrics import classification_report, roc_auc_score

def evaluate(model, test_loader, device):
    model.eval()

    correct = total = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, label, _ in test_loader:
            x, label = x.to(device), label.to(device)
            label_pred, _ = model(x)

            prob = torch.sigmoid(label_pred)
            preds = (prob > 0.5).float()

            correct += (preds == label).sum().item()
            total += label.size(0)

            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    print("Test Accuracy:", correct / total)
    print(classification_report(all_labels, [1 if p > 0.5 else 0 for p in all_probs]))
    print("ROC-AUC:", roc_auc_score(all_labels, all_probs))