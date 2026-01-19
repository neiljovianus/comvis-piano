import torch
import h5py
import numpy as np
import pretty_midi
import cv2
import os
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from bisect import bisect_right
from tqdm import tqdm

# Import Rumus Evaluasi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# BAGIAN 1: DEFINISI CLASS (HARUS SAMA PERSIS)
# ==========================================
# Kita copas Class Dataset & Model dari script training.
# Wajib sama biar file .pth bisa dibaca.

class R3FullDataset(Dataset):
    def __init__(self, hdf5_path, train_folder, fps=30, sequence_length=5):
        self.hdf5_path = hdf5_path
        self.train_folder = train_folder
        self.seq_len = sequence_length
        self.fps = fps
        self.segments = []
        self.total_frames = 0
        
        print(f"Scanning HDF5 groups in {os.path.basename(hdf5_path)}")
        with h5py.File(self.hdf5_path, 'r') as f:
            group_names = list(f.keys())
            group_names.sort()
            # Kita scan cepet aja tanpa ngecek satu2 biar loading eval cepet
            # Asumsi data evaluasi strukturnya sama kayak training
            for name in group_names:
                n_frames = f[name]['frames'].shape[0]
                valid_frames = n_frames - self.seq_len
                if valid_frames > 0:
                    start_idx = self.total_frames
                    end_idx = start_idx + valid_frames
                    # Note: Labels di-load lazy pas __getitem__ biar RAM hemat
                    self.segments.append({
                        'start': start_idx,
                        'end': end_idx,
                        'name': name
                    })
                    self.total_frames += valid_frames
        
        self.cumulative_indices = [s['end'] for s in self.segments]
        print(f"Loaded {len(self.segments)} splits. Total Samples: {self.total_frames}")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        segment_idx = bisect_right(self.cumulative_indices, idx)
        segment = self.segments[segment_idx]
        local_idx = idx - segment['start']
        
        # Load Data & Label on the fly
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[segment['name']]
            binary_frames = group['frames'][local_idx : local_idx + self.seq_len]
        
        # Logic Label MIDI (Disederhanakan buat eval, idealnya disimpen di hdf5 biar cepet)
        # Di sini kita perlu load label MIDI lagi. 
        # *Note: Agar script eval ini jalan, pastikan folder MIDI tersedia.*
        midi_filename = segment['name'] + ".midi"
        midi_path = os.path.join(self.train_folder, midi_filename)
        
        # Decode Images
        decoded_frames = []
        for bin_data in binary_frames:
            file_bytes = np.frombuffer(bin_data, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (640, 224)) 
            img = img.astype(np.float32) / 255.0
            decoded_frames.append(img)
        video_tensor = torch.from_numpy(np.array(decoded_frames)).unsqueeze(0)

        # Generate Label (Recalculate)
        # Idealnya label ini di-cache, tapi buat eval gapapa hitung ulang
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            label_vec = np.zeros(88, dtype=np.float32)
            current_time = (local_idx + self.seq_len - 1) / self.fps
            
            for note in midi_data.instruments[0].notes:
                if note.start <= current_time <= note.end:
                    key_idx = note.pitch - 21
                    if 0 <= key_idx < 88:
                        label_vec[key_idx] = 1.0
        except:
            label_vec = np.zeros(88, dtype=np.float32)

        label_tensor = torch.from_numpy(label_vec)
        return video_tensor, label_tensor

class VisualPianoModel(nn.Module):
    def __init__(self, num_keys=88, input_height=224, input_width=640):
        super().__init__()
        self.motion_conv = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        resnet = models.resnet18(weights=None) # Weights None karena kita mau load sendiri
        self.features = nn.Sequential(resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        # Kalkulasi manual flat size (sama kayak training)
        # Asumsi input size 224x640 -> output feat size 512 x 1 x 20 = 10240
        self.flat_size = 10240 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, num_keys)
        )

    def forward(self, x):
        motion = self.motion_conv(x).squeeze(2)
        last_frame = x[:, :, -1, :, :] 
        static = self.static_conv(last_frame)
        x = motion + static
        x = self.features(x)
        x = self.vertical_pool(x)
        x = self.classifier(x)
        return x

# ==========================================
# BAGIAN 2: LOGIC EVALUASI
# ==========================================

if __name__ == '__main__':
    # --- 1. SETTING PATH ---
    # Arahkan ke file model .pth yang mau dinilai
    # Ganti 'final_model_epoch1.pth' dengan nama file yang ada di folder checkpoints lo
    MODEL_PATH = 'checkpoints/final_model_epoch16.pth' 
    
    h5_file = 'processed_r3s/train/r3s_processed.hdf5'
    train_folder = 'processed_r3s/train/' 
    BATCH_SIZE = 32 # Bisa digedein dikit karena gak perlu simpen gradien (hemat VRAM)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧐 Evaluating on {DEVICE}")

    # --- 2. LOAD MODEL ---
    print(f"Loading Model from: {MODEL_PATH}")
    model = VisualPianoModel(num_keys=88, input_height=224, input_width=640)
    
    # Kunci: Load State Dict
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()

    model = model.to(DEVICE)
    model.eval() # Mode Ujian (Matikan Dropout/BatchNorm update)

    # --- 3. DATASET ---
    # Idealnya pake folder 'test', tapi kita pake sebagian 'train' dulu buat ngecek
    dataset = R3FullDataset(h5_file, train_folder, fps=30)
    
    # Ambil sampel kecil aja buat tes (misal 500 sampel pertama) biar gak nunggu seharian
    # Kalo mau full, hapus baris subset ini
    subset_indices = range(0, 1000) 
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Evaluating on {len(dataset)} samples...")

    # --- 4. TESTING LOOP ---
    all_preds = []
    all_targets = []
    
    print("Starting Inference...")
    with torch.no_grad(): # Matikan gradien (Wajib buat eval biar RAM lega)
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            
            # Forward
            outputs = model(inputs)
            
            # Convert Logits -> Probabilitas (Sigmoid)
            probs = torch.sigmoid(outputs)
            
            # Convert Probabilitas -> Biner (0 atau 1) pake Threshold 0.5
            preds = (probs > 0.5).float()
            
            # Simpan ke CPU buat dihitung sklearn
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    # Gabungin semua batch jadi satu array raksasa
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    
    # Flatten array biar jadi 1D list panjang (buat ngitung Global TP/TN/FP/FN)
    # Karena ini multi-label (88 keys), kita hitung "Micro Average" (Global Performance)
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # --- 5. HITUNG METRICS ---
    print("\n" + "="*40)
    print("📊 REPORT EVALUASI MODEL PIANO")
    print("="*40)
    
    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Confusion Matrix (TP, TN, FP, FN)
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()

    print(f"✅ Accuracy  : {acc:.4f}  (Seberapa sering bener tebak mati/nyala)")
    print(f"✅ Precision : {prec:.4f}  (Kalau nebak nyala, brp persen yg beneran nyala?)")
    print(f"✅ Recall    : {rec:.4f}  (Dari semua yg nyala, brp persen yg ketebak?)")
    print(f"✅ F1-Score  : {f1:.4f}   (Rata-rata harmonis Prec & Recall)")
    print("-" * 40)
    print("🔍 CONFUSION MATRIX (GLOBAL PIANO KEYS)")
    print(f"   [ TP (Benar Tekan) : {tp} ]   [ FP (Salah Tekan/Ghost) : {fp} ]")
    print(f"   [ FN (Gagal Deteksi): {fn} ]   [ TN (Benar Diam)        : {tn} ]")
    print("="*40)
    
    # Tips analisa
    print("\n💡 ANALISA SINGKAT:")
    if prec < rec:
        print("- Model 'Caper': Sering nebak tombol ditekan padahal enggak (Banyak Ghost Notes).")
    elif rec < prec:
        print("- Model 'Pemalu': Sering diem padahal ada tombol ditekan (Banyak Missed Notes).")
    else:
        print("- Model Seimbang!")