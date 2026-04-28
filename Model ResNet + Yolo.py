# IMPORTS
import os
import cv2
import time
import torch
import random
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from ultralytics import YOLO
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

# CONFIG
INPUT_DIR = "dataset/"
YOLO_ROOT = "dataset_yolo"
YOLO_IMAGES = os.path.join(YOLO_ROOT, "images")
YOLO_LABELS = os.path.join(YOLO_ROOT, "labels")
YOLO_TRAIN_IMAGES = os.path.join(YOLO_IMAGES, "train")
YOLO_VAL_IMAGES = os.path.join(YOLO_IMAGES, "val")
YOLO_TRAIN_LABELS = os.path.join(YOLO_LABELS, "train")
YOLO_VAL_LABELS = os.path.join(YOLO_LABELS, "val")

FINAL_DIR = "dataset_final"
BALANCED_DIR = "dataset_balanceado"

YOLO_RUN_PROJECT = "runs/detect"
YOLO_RUN_NAME = "train"
YOLO_MODEL_PATH = os.path.join(YOLO_RUN_PROJECT, YOLO_RUN_NAME, "weights", "best.pt")
YOLO_DATA_YAML = os.path.join(YOLO_ROOT, "dataset.yaml")

CONF_THRESHOLD = 0.50
YOLO_EPOCHS = 50
YOLO_IMGSZ = 640

RESNET_EPOCHS = 80
PATIENCE = 30
BATCH_SIZE = 32
RANDOM_STATE = 42

BEST_MODEL_PATH = "melhor_modelo.pth"
LAST_MODEL_PATH = "ultimo_modelo.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes_ruins = ["Germany", "Belgium", "Netherlands", "Canada", "Denmark"]

# HELPERS
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def is_image_file(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

def imread_safe(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_safe(path: str, img) -> bool:
    try:
        if img is None or getattr(img, "size", 0) == 0:
            return False
        ext = os.path.splitext(path)[1]
        if ext == "":
            ext = ".jpg"
            path = path + ext
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False

# 0. LIMPAR PASTAS GERADAS
print("Limpando pastas temporárias...")
reset_dir(YOLO_TRAIN_IMAGES)
reset_dir(YOLO_VAL_IMAGES)
reset_dir(YOLO_TRAIN_LABELS)
reset_dir(YOLO_VAL_LABELS)
reset_dir(FINAL_DIR)
reset_dir(BALANCED_DIR)

# 1. CRIAR DATASET YOLO (CLASSE FLAG)
print("Criando dataset YOLO para classe 'flag'...")

all_items = []
for classe in os.listdir(INPUT_DIR):
    class_dir = os.path.join(INPUT_DIR, classe)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        if is_image_file(img_name):
            all_items.append((classe, img_name))

train_items, val_items = train_test_split(
    all_items,
    test_size=0.2,
    random_state=RANDOM_STATE
)

def copy_with_full_flag_box(items, img_out_dir, lbl_out_dir):
    for classe, img_name in items:
        src = os.path.join(INPUT_DIR, classe, img_name)
        if not os.path.exists(src):
            continue

        img = imread_safe(src)
        if img is None:
            print(f"Não foi possível ler: {src}")
            continue

        new_name = f"{classe}_{img_name}"
        dst_img = os.path.join(img_out_dir, new_name)
        ok = imwrite_safe(dst_img, img)
        if not ok:
            print(f"Não foi possível salvar: {dst_img}")
            continue

        stem = os.path.splitext(new_name)[0]
        dst_lbl = os.path.join(lbl_out_dir, f"{stem}.txt")

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 1.0 1.0\n")

copy_with_full_flag_box(train_items, YOLO_TRAIN_IMAGES, YOLO_TRAIN_LABELS)
copy_with_full_flag_box(val_items, YOLO_VAL_IMAGES, YOLO_VAL_LABELS)

yaml_text = f"""path: {YOLO_ROOT}
train: images/train
val: images/val

names:
  0: flag
"""
with open(YOLO_DATA_YAML, "w", encoding="utf-8") as f:
    f.write(yaml_text)

print("Dataset YOLO criado.")
print(f"Treino YOLO: {len(train_items)} imagens")
print(f"Val YOLO: {len(val_items)} imagens")

# 2. TREINAR YOLO FLAG
print("Treinando YOLO específico para bandeiras...")

yolo_train_model = YOLO("yolov8n.pt")
yolo_train_model.train(
    data=YOLO_DATA_YAML,
    epochs=YOLO_EPOCHS,
    imgsz=YOLO_IMGSZ,
    project=YOLO_RUN_PROJECT,
    name=YOLO_RUN_NAME,
    exist_ok=True
)

print("YOLO treinado.")

# 3. VALIDAR YOLO E PEGAR mAP
print("Validando YOLO...")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_val_results = yolo_model.val(data=YOLO_DATA_YAML, imgsz=YOLO_IMGSZ, verbose=False)

yolo_map50 = None
yolo_map50_95 = None

try:
    if hasattr(yolo_val_results, "box") and yolo_val_results.box is not None:
        if hasattr(yolo_val_results.box, "map50"):
            yolo_map50 = float(yolo_val_results.box.map50)
        if hasattr(yolo_val_results.box, "map"):
            yolo_map50_95 = float(yolo_val_results.box.map)
except Exception:
    pass

# 4. VISUALIZAR EXEMPLOS DO YOLO TREINADO
def visualizar_exemplos_yolo(pasta_imagens, n=5):
    print(f"\nMostrando {n} exemplos com bounding boxes do YOLO treinado...")
    imagens = [x for x in os.listdir(pasta_imagens) if is_image_file(x)]
    if not imagens:
        print("Nenhuma imagem encontrada.")
        return

    amostra = random.sample(imagens, min(n, len(imagens)))

    for img_name in amostra:
        img_path = os.path.join(pasta_imagens, img_name)
        img = imread_safe(img_path)
        if img is None:
            print(f"Falha ao abrir: {img_path}")
            continue

        resultados = yolo_model(img, verbose=False)

        for r in resultados:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(img_name)
        plt.axis("off")
        plt.show()

visualizar_exemplos_yolo(YOLO_VAL_IMAGES, n=5)

# 5. GERAR DATASET FINAL COM O YOLO TREINADO
print("Gerando dataset final a partir do YOLO treinado...")

for classe in os.listdir(INPUT_DIR):
    in_path = os.path.join(INPUT_DIR, classe)
    out_path = os.path.join(FINAL_DIR, classe)
    ensure_dir(out_path)

    if not os.path.isdir(in_path):
        continue

    for img_name in os.listdir(in_path):
        if not is_image_file(img_name):
            continue

        img_path = os.path.join(in_path, img_name)
        img = imread_safe(img_path)
        if img is None:
            print(f"Não foi possível ler: {img_path}")
            continue

        results = yolo_model(img, verbose=False)

        best_box = None
        best_conf = -1.0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = box

        if best_box is None:
            continue

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        out_name = f"{os.path.splitext(img_name)[0]}_crop.jpg"
        out_file = os.path.join(out_path, out_name)
        ok = imwrite_safe(out_file, crop)
        if not ok:
            print(f"Falha ao salvar crop: {out_file}")

print("Dataset final criado com o YOLO treinado.")

# 6. VISUALIZAÇÃO AUTOMÁTICA DO DATASET FINAL
def visualizar_exemplos_dataset_final(root_dir, n_por_classe=2):
    print("\nVisualizando exemplos do dataset final...")

    for classe in os.listdir(root_dir):
        classe_dir = os.path.join(root_dir, classe)
        if not os.path.isdir(classe_dir):
            continue

        imagens = [x for x in os.listdir(classe_dir) if is_image_file(x)]
        if not imagens:
            continue

        amostra = random.sample(imagens, min(n_por_classe, len(imagens)))

        for img_name in amostra:
            img_path = os.path.join(classe_dir, img_name)
            img = imread_safe(img_path)
            if img is None:
                continue

            plt.figure(figsize=(4, 4))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"{classe} | {img_name}")
            plt.axis("off")
            plt.show()

visualizar_exemplos_dataset_final(FINAL_DIR, n_por_classe=2)

# 7. DATASET BALANCEADO
print("Balanceando dataset...")

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

for classe in os.listdir(FINAL_DIR):
    in_path = os.path.join(FINAL_DIR, classe)
    out_path = os.path.join(BALANCED_DIR, classe)
    ensure_dir(out_path)

    if not os.path.isdir(in_path):
        continue

    for img_name in os.listdir(in_path):
        if not is_image_file(img_name):
            continue

        img_path = os.path.join(in_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"Falha ao abrir com PIL: {img_path}")
            continue

        img.save(os.path.join(out_path, img_name))

        if classe in classes_ruins:
            n_aug, aug = 5, augment_forte
        else:
            n_aug, aug = 1, augment_leve

        for i in range(n_aug):
            aug_img = aug(img)
            aug_img.save(os.path.join(out_path, f"aug_{i}_{img_name}"))

print("Dataset balanceado pronto!")

# 8. DATASET + SPLIT ESTRATIFICADO
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.6, 1.0)),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomHorizontalFlip(0.5),
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

full_dataset = datasets.ImageFolder(root=BALANCED_DIR)
classes = full_dataset.classes
targets = full_dataset.targets
indices = list(range(len(full_dataset)))

train_idx, temp_idx, train_y, temp_y = train_test_split(
    indices,
    targets,
    test_size=0.30,
    stratify=targets,
    random_state=RANDOM_STATE
)

val_idx, test_idx, _, _ = train_test_split(
    temp_idx,
    temp_y,
    test_size=0.50,
    stratify=temp_y,
    random_state=RANDOM_STATE
)

train_base = datasets.ImageFolder(root=BALANCED_DIR, transform=train_transform)
val_base = datasets.ImageFolder(root=BALANCED_DIR, transform=val_transform)
test_base = datasets.ImageFolder(root=BALANCED_DIR, transform=val_transform)

train_dataset = Subset(train_base, train_idx)
val_dataset = Subset(val_base, val_idx)
test_dataset = Subset(test_base, test_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Treino:", len(train_dataset))
print("Validação:", len(val_dataset))
print("Teste:", len(test_dataset))

# 9. MODELO RESNET
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

# 10. TREINO + EARLY STOPPING
train_losses = []
val_accuracies = []

best_val = -1.0
best_epoch = -1
counter = 0
min_delta = 1e-6

print("Treinando ResNet...")

for epoch in range(RESNET_EPOCHS):
    model.train()
    total_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            out = model(imgs)
            _, preds = torch.max(out, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total if total > 0 else 0.0

    train_losses.append(total_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1} | Loss {total_loss:.4f} | Val {val_acc:.2f}%")
    
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

# 11. GRÁFICO
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Loss")
plt.plot(val_accuracies, label="Val Accuracy")
plt.title("Treinamento")
plt.legend()
plt.show()

# 12. RESULTADOS RESNET
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
classes = checkpoint["classes"]

print(f"Época salva: {checkpoint.get('epoch', 'N/A')}")
print(f"Melhor Val Acc: {checkpoint.get('best_val_acc', 'N/A')}")

model.eval()

all_preds = []
all_labels = []
all_probs = []

# tempo de inferência
total_inference_time = 0.0
total_images = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        start_time = time.perf_counter()

        out = model(imgs)
        probs = torch.softmax(out, dim=1)
        _, preds = torch.max(out, 1)

        end_time = time.perf_counter()

        total_inference_time += (end_time - start_time)
        total_images += imgs.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

avg_inference_time = total_inference_time / total_images if total_images > 0 else 0.0

print(f"\nTempo médio de inferência por imagem: {avg_inference_time:.6f} s")

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

labels_ids = list(range(len(classes)))

accuracy = accuracy_score(all_labels, all_preds)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    labels=labels_ids,
    average="macro",
    zero_division=0
)

precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    labels=labels_ids,
    average="weighted",
    zero_division=0
)

y_true_bin = label_binarize(all_labels, classes=labels_ids)
try:
    map_classification_macro = average_precision_score(
        y_true_bin,
        all_probs,
        average="macro"
    )
except ValueError:
    map_classification_macro = None

print("\nMÉTRICAS GERAIS DO RESNET")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão (macro): {precision_macro:.4f}")
print(f"Recall (macro): {recall_macro:.4f}")
print(f"F1-score (macro): {f1_macro:.4f}")
print(f"Precisão (weighted): {precision_weighted:.4f}")
print(f"Recall (weighted): {recall_weighted:.4f}")
print(f"F1-score (weighted): {f1_weighted:.4f}")

if map_classification_macro is not None:
    print(f"mAP classificação (macro AP): {map_classification_macro:.4f}")
else:
    print("mAP classificação (macro AP): não pôde ser calculado")

print("\nRELATÓRIO FINAL")
print(classification_report(
    all_labels,
    all_preds,
    labels=labels_ids,
    target_names=classes,
    zero_division=0
))

cm = confusion_matrix(all_labels, all_preds, labels=labels_ids)

plt.figure(figsize=(10, 8))
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

# 13. mAP DO YOLO
print("\nMÉTRICAS DO YOLO")
if yolo_map50 is not None:
    print(f"YOLO mAP@0.50: {yolo_map50:.4f}")
else:
    print("YOLO mAP@0.50: não disponível")

if yolo_map50_95 is not None:
    print(f"YOLO mAP@0.50:0.95: {yolo_map50_95:.4f}")
else:
    print("YOLO mAP@0.50:0.95: não disponível")

# 14. INFERÊNCIA FINAL YOLO + RESNET COM TOP-3
def detectar_e_classificar(img_path):
    img = imread_safe(img_path)
    if img is None:
        print("Erro ao carregar imagem.")
        return None

    start_time = time.perf_counter()

    results = yolo_model(img, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            tensor = val_transform(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)

            top_probs, top_idxs = torch.topk(probs, 3)

            top_texts = []
            for rank in range(min(3, top_idxs.shape[1])):
                cls_name = classes[top_idxs[0, rank].item()]
                cls_prob = top_probs[0, rank].item() * 100
                top_texts.append(f"{rank+1}: {cls_name} ({cls_prob:.1f}%)")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            base_y = max(20, y1 - 10)
            for i, txt in enumerate(top_texts):
                y_text = base_y + i * 25
                cv2.putText(
                    img,
                    txt,
                    (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

    inference_time = time.perf_counter() - start_time
    print(f"\nTempo de inferência: {inference_time:.4f} s")

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("YOLO + ResNet (Top-3)")
    plt.show()

    return inference_time

# 15. COMPARAÇÃO MELHOR x ÚLTIMO
best_ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
last_ckpt = torch.load(LAST_MODEL_PATH, map_location=device)

print("\nCOMPARAÇÃO DOS CHECKPOINTS")
print("Melhor modelo -> época:", best_ckpt.get("epoch"), "| val_acc:", best_ckpt.get("best_val_acc"))
print("Último modelo -> época:", last_ckpt.get("epoch"), "| val_acc:", last_ckpt.get("val_acc"))

# 16. RESUMO FINAL
print("\n==================== RESUMO FINAL ====================")
print(f"Melhor época ResNet: {best_epoch}")
print(f"Melhor Val Acc ResNet: {best_val:.4f}")
print(f"ResNet Accuracy: {accuracy:.4f}")
print(f"ResNet Precision (macro): {precision_macro:.4f}")
print(f"ResNet Recall (macro): {recall_macro:.4f}")
print(f"ResNet F1-score (macro): {f1_macro:.4f}")
print(f"Tempo médio de inferência por imagem: {avg_inference_time:.6f} s")
if map_classification_macro is not None:
    print(f"ResNet mAP (classificação): {map_classification_macro:.4f}")
else:
    print("ResNet mAP (classificação): não disponível")

if yolo_map50 is not None:
    print(f"YOLO mAP@0.50: {yolo_map50:.4f}")
else:
    print("YOLO mAP@0.50: não disponível")

if yolo_map50_95 is not None:
    print(f"YOLO mAP@0.50:0.95: {yolo_map50_95:.4f}")
else:
    print("YOLO mAP@0.50:0.95: não disponível")
print("=====================================================")
