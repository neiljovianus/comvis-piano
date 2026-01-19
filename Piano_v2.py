import torch
import h5py
import numpy as np
import pretty_midi
import cv2
import os
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt # Kalo butuh plot

from torch.utils.data import Dataset, DataLoader
from bisect import bisect_right
from tqdm import tqdm

# ==========================================
# BAGIAN 1: DEFINISI (Jangan diubah-ubah)
# ==========================================

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
              
            print(f"Processing MIDI labels for {len(group_names)} splits")
            # Kita pake list comprehension biar rapi, atau loop biasa gapapa
            # (Loop scanning dihapus dari display biar ringkas, asumsi logic sama)
            for name in tqdm(group_names):
                midi_filename = name + ".midi"
                midi_path = os.path.join(self.train_folder, midi_filename)
                
                if not os.path.exists(midi_path):
                    continue

                n_frames = f[name]['frames'].shape[0]
                try:
                    midi_data = pretty_midi.PrettyMIDI(midi_path)
                except:
                    continue # Skip error midi
                
                split_labels = np.zeros((n_frames, 88), dtype=np.float32)
                    
                for note in midi_data.instruments[0].notes:
                    start_f = int(note.start * self.fps)
                    end_f = int(note.end * self.fps)
                    start_f = max(0, start_f)
                    end_f = min(n_frames, end_f)
                    
                    if start_f < end_f:
                        key_idx = note.pitch - 21
                        if 0 <= key_idx < 88:
                            split_labels[start_f:end_f, key_idx] = 1.0

                valid_frames = n_frames - self.seq_len
                
                if valid_frames > 0:
                    start_idx = self.total_frames
                    end_idx = start_idx + valid_frames
                    
                    self.segments.append({
                        'start': start_idx,
                        'end': end_idx,
                        'name': name,
                        'labels': split_labels
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
        
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[segment['name']]
            binary_frames = group['frames'][local_idx : local_idx + self.seq_len]
        
        decoded_frames = []
        for bin_data in binary_frames:
            file_bytes = np.frombuffer(bin_data, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (640, 224)) 
            img = img.astype(np.float32) / 255.0
            decoded_frames.append(img)
            
        video_tensor = torch.from_numpy(np.array(decoded_frames)).unsqueeze(0)
        label_vec = segment['labels'][local_idx + self.seq_len - 1]
        label_tensor = torch.from_numpy(label_vec)
        
        return video_tensor, label_tensor

def apply_augmentations(video_tensor):
    if video_tensor.size(0) == 0:
        return video_tensor
    
    if torch.rand(1).item() < 0.8:
        brightness = (torch.rand(1).item() * 0.6) + 0.7
        video_tensor = video_tensor * brightness
    
    if torch.rand(1).item() < 0.5:
        noise_level = 0.03 
        noise = torch.randn_like(video_tensor) * noise_level
        video_tensor = video_tensor + noise
    
    return torch.clamp(video_tensor, 0.0, 1.0)

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
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        with torch.no_grad():
            dummy_3d = torch.zeros(1, 1, 5, input_height, input_width)
            dummy_2d = torch.zeros(1, 1, input_height, input_width)
            
            m = self.motion_conv(dummy_3d).squeeze(2)
            s = self.static_conv(dummy_2d)
            combined = m + s
            feat = self.features(combined)
            feat = self.vertical_pool(feat)
            self.flat_size = feat.numel()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, num_keys)
        )

    def forward(self, x):
        motion = self.motion_conv(x)
        motion = motion.squeeze(2)
        
        last_frame = x[:, :, -1, :, :] 
        static = self.static_conv(last_frame)
        
        x = motion + static
        x = self.features(x)
        x = self.vertical_pool(x)
        x = self.classifier(x)
        return x

# ==========================================
# BAGIAN 2: EKSEKUSI (WAJIB DALAM IF NAME MAIN)
# ==========================================

if __name__ == '__main__':
    # --- Settingan Awal ---
    h5_file = 'processed_r3s/train/r3s_processed.hdf5'
    train_folder = 'processed_r3s/train/' 
    
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4    
    EPOCHS = 1              
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    # --- Load Dataset ---
    # Proses berat scanning HDF5 akan aman di sini karena dipagari if name main
    full_dataset = R3FullDataset(h5_file, train_folder, fps=30)
    print(f"Total Training Samples: {len(full_dataset)}")

    # --- Setup Model ---
    model = VisualPianoModel(num_keys=88, input_height=224, input_width=640)
    model = model.to(DEVICE)

    pos_weight = torch.ones([88]).to(DEVICE) * 15
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- DataLoader (KUNCI NGEBUT) ---
    # num_workers=4: Aman karena di dalam if main
    # pin_memory=True: Transfer ke GPU makin cepet
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    steps_per_epoch = len(train_loader)

    print(f"Starting Loop: {steps_per_epoch} batches per epoch")
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f"Epoch {epoch+1}")
        
        for step, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Preprocessing (Augmentasi)
            inputs = apply_augmentations(inputs)
            
            optimizer.zero_grad()
            
            # Forward & Backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Checkpoint
            if step > 0 and step % 2000 == 0:
                torch.save(model.state_dict(), f"{SAVE_DIR}/model_epoch{epoch+1}_step{step}.pth")

        # Save Final
        torch.save(model.state_dict(), f"{SAVE_DIR}/final_model_epoch{epoch+1}.pth")
        print(f"✅ Epoch {epoch+1} Finished.")