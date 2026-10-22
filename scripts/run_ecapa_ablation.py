import os
import sys
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.stdout.reconfigure(encoding='utf-8')

device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print(f"Executing on device: {device_str}")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_raw = os.path.join(base_dir, "data", "05_speaker_dataset", "raw")
dir_denoised = os.path.join(base_dir, "data", "05_speaker_dataset", "denoised")

speakers = ["吳", "甯", "白", "蕭"]
spk2idx = {s: i for i, s in enumerate(speakers)}
idx2spk = {i: s for i, s in enumerate(speakers)}

print("Loading SpeechBrain ECAPA-TDNN pretrained backbone...")
ecapa_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device_str}
)

def collect_files(data_dir):
    records = []
    for spk in speakers:
        spk_dir = os.path.join(data_dir, spk)
        for fn in sorted(os.listdir(spk_dir)):
            if fn.endswith('.wav'):
                fp = os.path.join(spk_dir, fn)
                records.append((fp, spk2idx[spk], spk, fn))
    return records

class AudioDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, label, spk, fn = self.records[idx]
        wav, sr = sf.read(fp)
        return torch.tensor(wav, dtype=torch.float32), label, spk, fn

def extract_all_embeddings(records):
    embeddings = []
    labels = []
    with torch.no_grad():
        for fp, label, spk, fn in records:
            wav, sr = sf.read(fp)
            wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
            # ECAPA-TDNN encode_batch returns (batch, 1, 192)
            emb = ecapa_model.encode_batch(wav_tensor).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm > 1e-7:
                emb = emb / norm
            embeddings.append(emb)
            labels.append(label)
    return np.array(embeddings), np.array(labels)

def compute_separation_metrics(embeddings, labels):
    # Cosine similarities
    sim_matrix = np.dot(embeddings, embeddings.T)
    n = len(labels)
    intra_sims = []
    inter_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if labels[i] == labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)
    intra_mean = float(np.mean(intra_sims))
    inter_mean = float(np.mean(inter_sims))
    margin = intra_mean - inter_mean
    sil = float(silhouette_score(embeddings, labels, metric='cosine'))
    return {
        'intra_sim': intra_mean,
        'inter_sim': inter_mean,
        'margin': margin,
        'silhouette': sil
    }

class ECAPAClassifier(nn.Module):
    def __init__(self, emb_dim=192, num_classes=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        return self.fc(x)

def train_and_eval(train_records, test_records, num_epochs=25, lr=1e-3):
    # Pre-extract embeddings for fast training of the classifier head
    train_embs, train_y = extract_all_embeddings(train_records)
    test_embs, test_y = extract_all_embeddings(test_records)
    
    train_x_t = torch.tensor(train_embs, dtype=torch.float32).to(device)
    train_y_t = torch.tensor(train_y, dtype=torch.long).to(device)
    test_x_t = torch.tensor(test_embs, dtype=torch.float32).to(device)
    test_y_t = torch.tensor(test_y, dtype=torch.long).to(device)
    
    model = ECAPAClassifier(emb_dim=192, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = []
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(train_x_t)
        loss = criterion(out, train_y_t)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(out, dim=1).cpu().numpy()
        train_acc = accuracy_score(train_y, preds)
        
        model.eval()
        with torch.no_grad():
            test_out = model(test_x_t)
            test_loss = criterion(test_out, test_y_t).item()
            test_preds = torch.argmax(test_out, dim=1).cpu().numpy()
            test_acc = accuracy_score(test_y, test_preds)
            test_f1 = f1_score(test_y, test_preds, average='macro')
            
        history.append({
            'epoch': epoch,
            'train_loss': loss.item(),
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        })
        
    cm = confusion_matrix(test_y, test_preds)
    return model, history, test_acc, test_f1, cm

def run_experiment(data_dir, condition_name):
    print(f"\n=======================================================")
    print(f"   RUNNING EXPERIMENT: {condition_name}")
    print(f"=======================================================")
    records = collect_files(data_dir)
    print(f"Collected {len(records)} total audio files across {len(speakers)} speakers.")
    
    # 1. Zero-shot Embedding Quality
    all_embs, all_labels = extract_all_embeddings(records)
    sep_metrics = compute_separation_metrics(all_embs, all_labels)
    print(f"[ECAPA-TDNN Embedding Metrics]")
    print(f"  Intra-speaker Cosine Similarity: {sep_metrics['intra_sim']:.4f} (higher is better)")
    print(f"  Inter-speaker Cosine Similarity: {sep_metrics['inter_sim']:.4f} (lower is better)")
    print(f"  Separation Margin (Intra - Inter): {sep_metrics['margin']:.4f} (higher is better)")
    print(f"  Silhouette Score (Cosine Space):  {sep_metrics['silhouette']:.4f} (higher is better)")
    
    # 2. Stratified Train / Test Split (70% train, 30% test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    labels = [r[1] for r in records]
    train_idx, test_idx = next(sss.split(records, labels))
    train_recs = [records[i] for i in train_idx]
    test_recs = [records[i] for i in test_idx]
    
    print(f"Split: {len(train_recs)} Train samples, {len(test_recs)} Test samples")
    
    # 3. Train Model
    trained_model, history, final_acc, final_f1, cm = train_and_eval(train_recs, test_recs, num_epochs=30)
    
    print(f"[Classification Results @ Epoch 30]")
    print(f"  Final Test Accuracy: {final_acc * 100:.2f}%")
    print(f"  Final Test Macro F1: {final_f1 * 100:.2f}%")
    print(f"  Confusion Matrix (Rows: True, Cols: Pred):\n{cm}")
    
    # Save checkpoint
    save_path = os.path.join(base_dir, "models", f"ecapa_classifier_{condition_name}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    
    return {
        'condition': condition_name,
        'metrics': sep_metrics,
        'final_acc': final_acc,
        'final_f1': final_f1,
        'confusion_matrix': cm,
        'history': history
    }

# Run Ablation: Condition A (未降噪) vs Condition B (有降噪)
res_raw = run_experiment(dir_raw, "未降噪_Raw")
res_denoised = run_experiment(dir_denoised, "有降噪_Denoised")

print("\n" + "="*70)
print("             ABLATION STUDY COMPARISON SUMMARY               ")
print("="*70)
print(f"{'Metric':<32} | {'Condition A (未降噪)':<20} | {'Condition B (有降噪)':<20}")
print("-" * 76)
print(f"{'Intra-speaker Similarity':<32} | {res_raw['metrics']['intra_sim']:<20.4f} | {res_denoised['metrics']['intra_sim']:<20.4f}")
print(f"{'Inter-speaker Similarity':<32} | {res_raw['metrics']['inter_sim']:<20.4f} | {res_denoised['metrics']['inter_sim']:<20.4f}")
print(f"{'Separation Margin (Intra-Inter)':<32} | {res_raw['metrics']['margin']:<20.4f} | {res_denoised['metrics']['margin']:<20.4f}")
print(f"{'Cosine Silhouette Score':<32} | {res_raw['metrics']['silhouette']:<20.4f} | {res_denoised['metrics']['silhouette']:<20.4f}")
print(f"{'Test Classification Accuracy':<32} | {res_raw['final_acc']*100:<19.2f}% | {res_denoised['final_acc']*100:<19.2f}%")
print(f"{'Test Macro F1-Score':<32} | {res_raw['final_f1']*100:<19.2f}% | {res_denoised['final_f1']*100:<19.2f}%")
print("="*70)
