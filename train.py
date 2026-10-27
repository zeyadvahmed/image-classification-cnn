"""Image classification pipeline: data prep -> CNN -> evaluation -> Grad-CAM.
Dataset: scikit-learn digits (8x8 grayscale, 10 classes), used as an offline stand-in.
Swap `load_data()` for a medical dataset (e.g. chest X-ray folders) to reuse the same pipeline.
"""
import json, base64, io
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def load_data():
    d = load_digits()
    X = (d.images / 16.0).astype("float32")[:, None]   # N,1,8,8
    return X, d.target


class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)

    def features(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        return F.relu(self.c3(x))

    def forward(self, x):
        f = self.features(x)
        return self.fc(self.drop(f.mean((2, 3))))


def augment(x):
    # random shift up to 1px + small noise
    dx, dy = np.random.randint(-1, 2, 2)
    x = torch.roll(x, shifts=(int(dx), int(dy)), dims=(2, 3))
    return x + 0.02 * torch.randn_like(x)


def png_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


def gradcam(model, x, cls):
    model.eval(); x = x.clone().requires_grad_(True)
    f = model.features(x); f.retain_grad()
    out = model.fc(f.mean((2, 3)))
    out[0, cls].backward()
    w = f.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * f).sum(1))[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def main():
    X, y = load_data()
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=SEED)
    Xtr, Xva, Xte = map(torch.tensor, (Xtr, Xva, Xte))
    ytr_t, yva_t = torch.tensor(ytr), torch.tensor(yva)

    model = CNN()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    best, best_state, patience, hist = 0, None, 0, []
    for ep in range(40):
        model.train(); perm = torch.randperm(len(Xtr)); tl = 0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(model(augment(Xtr[idx])), ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        sched.step(); model.eval()
        with torch.no_grad():
            va = (model(Xva).argmax(1) == yva_t).float().mean().item()
        hist.append((tl / len(Xtr), va))
        if va > best: best, best_state, patience = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8: break
    model.load_state_dict(best_state); model.eval()
    torch.save(best_state, "model.pt")

    with torch.no_grad():
        prob = F.softmax(model(Xte), 1).numpy()
    pred = prob.argmax(1)
    acc = accuracy_score(yte, pred)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    auc = roc_auc_score(yte, prob, multi_class="ovr")
    cm = confusion_matrix(yte, pred)
    spec = []
    for c in range(10):
        tn = cm.sum() - cm[c].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        spec.append(tn / (tn + fp))
    sens = [cm[c, c] / cm[c].sum() for c in range(10)]

    # figures
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "#333")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    cm_img = png_b64(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.plot([h[0] for h in hist], label="train loss"); ax2 = ax.twinx()
    ax2.plot([h[1] for h in hist], color="tab:orange", label="val acc")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax2.set_ylabel("val accuracy")
    curve_img = png_b64(fig)

    fig, axs = plt.subplots(2, 6, figsize=(9, 3.4))
    Xte_np = Xte.numpy()
    for k in range(6):
        i = k * 7
        cam = gradcam(model, Xte[i:i + 1], int(pred[i]))
        axs[0, k].imshow(Xte_np[i, 0], cmap="gray"); axs[0, k].set_title(f"true {yte[i]} / pred {pred[i]}", fontsize=8)
        axs[1, k].imshow(Xte_np[i, 0], cmap="gray")
        axs[1, k].imshow(np.kron(cam, np.ones((1, 1))), cmap="jet", alpha=0.5)
        for a in (axs[0, k], axs[1, k]): a.axis("off")
    gc_img = png_b64(fig)

    res = dict(n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte), epochs=len(hist),
               accuracy=acc, precision=p, recall=r, f1=f1, auc=auc,
               sens=[float(s) for s in sens], spec=[float(s) for s in spec],
               cm_img=cm_img, curve_img=curve_img, gc_img=gc_img)
    json.dump(res, open("results.json", "w"))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if "img" not in k and k not in ("sens", "spec")})
    print("mean sens", np.mean(sens), "mean spec", np.mean(spec))


if __name__ == "__main__":
    main()
