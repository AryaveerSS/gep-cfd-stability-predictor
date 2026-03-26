import torch
from sklearn.metrics import classification_report, roc_auc_score

def evaluate(model, test_loader, device):
    model.eval()

    correct = total = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for token_ids, depth_ids, subtree_ids, label, _ in test_loader:
            token_ids   = token_ids.to(device)
            depth_ids   = depth_ids.to(device)
            subtree_ids = subtree_ids.to(device)
            label       = label.to(device)

            label_pred, _ = model(token_ids, depth_ids, subtree_ids)

            prob = torch.sigmoid(label_pred)
            preds = (prob > 0.5).float()

            correct += (preds == label).sum().item()
            total += label.size(0)

            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    print("Test Accuracy:", correct / total)
    print(classification_report(all_labels, [1 if p > 0.5 else 0 for p in all_probs]))
    print("ROC-AUC:", roc_auc_score(all_labels, all_probs))