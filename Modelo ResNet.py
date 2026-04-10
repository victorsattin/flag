# IMPORTS
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# 1. GERAR DATASET BALANCEADO
input_dir = "dataset/"
output_dir = "dataset balanceado/"

classes_ruins = [
    "Germany",
    "Belgium",
    "Netherlands",
    "Canada",
    "Denmark"
]

if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:

    print("Criando dataset balanceado...")

    augment_leve = transforms.Compose([
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomRotation(10)
    ])

    augment_forte = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.6, 1.0)),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.RandomHorizontalFlip(0.5)
    ])

    for classe in os.listdir(input_dir):
        in_path = os.path.join(input_dir, classe)
        out_path = os.path.join(output_dir, classe)

        os.makedirs(out_path, exist_ok=True)

        for img_name in os.listdir(in_path):
            img_path = os.path.join(in_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")

                img.save(os.path.join(out_path, img_name))

                if classe in classes_ruins:
                    n_aug = 5
                    transform_aug = augment_forte
                else:
                    n_aug = 1
                    transform_aug = augment_leve

                for i in range(n_aug):
                    aug_img = transform_aug(img)
                    aug_img.save(os.path.join(out_path, f"aug_{i}_{img_name}"))

            except:
                pass

    print("Dataset balanceado criado!")

# 2. CONFIG TREINO
data_dir = "dataset balanceado/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 40
PATIENCE = 10
BATCH_SIZE = 32

BEST_MODEL_PATH = "melhor_modelo_resnet.pth"
LAST_MODEL_PATH = "ultimo_modelo_resnet.pth"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.6, 1.0)),
    transforms.RandomRotation(25),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3. DATASET
full_dataset = datasets.ImageFolder(root=data_dir)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = full_dataset.classes

# 4. MODELO
model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0003
)

train_losses = []
val_accuracies = []

best_val = -1.0
best_epoch = -1
counter = 0
min_delta = 1e-6

print("Iniciando treinamento...")

# 5. TREINO + EARLY STOPPING
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total if total > 0 else 0.0

    train_losses.append(total_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "classes": classes,
        "val_acc": val_acc
    }, LAST_MODEL_PATH)

    if val_acc > best_val + min_delta:
        best_val = val_acc
        best_epoch = epoch + 1
        counter = 0

        torch.save({
            "epoch": best_epoch,
            "model_state": model.state_dict(),
            "classes": classes,
            "best_val_acc": best_val
        }, BEST_MODEL_PATH)

        print(f"Melhor modelo salvo! Época: {best_epoch} | Val Acc: {best_val:.2f}%")
    else:
        counter += 1
        print(f"Sem melhora. Paciência: {counter}/{PATIENCE}")

    if counter >= PATIENCE:
        print("Early stopping")
        break

print(f"\nMelhor época: {best_epoch} | Melhor Val Acc: {best_val:.2f}%")

# 6. GRÁFICO
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Loss")
plt.plot(val_accuracies, label="Val Accuracy")
plt.title("Treinamento")
plt.legend()
plt.show()

# 7. RESULTADOS FINAIS
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
classes = checkpoint["classes"]

print(f"Época salva: {checkpoint.get('epoch', 'N/A')}")
print(f"Melhor Val Acc: {checkpoint.get('best_val_acc', 'N/A')}")

model.eval()

all_preds = []
all_labels = []

total_inference_time = 0.0
total_images = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        start_time = time.perf_counter()

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        end_time = time.perf_counter()

        total_inference_time += (end_time - start_time)
        total_images += images.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

avg_inference_time = total_inference_time / total_images if total_images > 0 else 0.0

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0
)

print("\nMÉTRICAS GERAIS")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão (macro): {precision_macro:.4f}")
print(f"Recall (macro): {recall_macro:.4f}")
print(f"F1-score (macro): {f1_macro:.4f}")
print(f"Precisão (weighted): {precision_weighted:.4f}")
print(f"Recall (weighted): {recall_weighted:.4f}")
print(f"F1-score (weighted): {f1_weighted:.4f}")
print(f"Tempo médio de inferência por imagem: {avg_inference_time:.6f} s")

print("\nRELATÓRIO FINAL")
print(classification_report(
    all_labels,
    all_preds,
    target_names=classes,
    zero_division=0
))

# 8. MATRIZ DE CONFUSÃO
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)

plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# 9. COMPARAÇÃO MELHOR x ÚLTIMO
best_ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
last_ckpt = torch.load(LAST_MODEL_PATH, map_location=device)

print("\nCOMPARAÇÃO DOS CHECKPOINTS")
print("Melhor modelo -> época:", best_ckpt.get("epoch"), "| val_acc:", best_ckpt.get("best_val_acc"))
print("Último modelo -> época:", last_ckpt.get("epoch"), "| val_acc:", last_ckpt.get("val_acc"))

# 10. RESUMO FINAL
print("\n==================== RESUMO FINAL ====================")
print(f"Melhor época ResNet: {best_epoch}")
print(f"Melhor Val Acc ResNet: {best_val:.4f}")
print(f"ResNet Accuracy: {accuracy:.4f}")
print(f"ResNet Precision (macro): {precision_macro:.4f}")
print(f"ResNet Recall (macro): {recall_macro:.4f}")
print(f"ResNet F1-score (macro): {f1_macro:.4f}")
print(f"Tempo médio de inferência por imagem: {avg_inference_time:.6f} s")
print("=====================================================")