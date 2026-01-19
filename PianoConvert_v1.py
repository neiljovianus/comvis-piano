import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# 1. STRUKTUR MODEL (Copas dari script lu)
#    Gw bersihin import sampah, sisa torch doang
# ==========================================

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
        
        # PRETRAINED BACKBONE
        # Note: Pastiin internet nyala bentar buat load struktur resnet default, 
        # tapi nanti bobotnya bakal ketimpa sama PTH lu kok. Aman.
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
# 2. LOGIC CONVERT (PTH -> PT)
# ==========================================

if __name__ == '__main__':
    # --- KONFIGURASI ---
    PTH_PATH = "./checkpoints/final_model_epoch16.pth" # <--- GANTI SESUAI LOKASI FILE LU
    PT_OUTPUT = "piano_final_model.pt"
    # Shape Input Model Lu: (Batch, Channel, Depth/Frames, Height, Width)
    # Channel = 1 (Grayscale), Depth = 5 (Seq Length)
    INPUT_SHAPE = (1, 1, 5, 224, 640) 
    # -------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Running on: {device}")

    try:
        print("1️⃣  Inisialisasi Model Kosong...")
        model = VisualPianoModel(num_keys=88, input_height=224, input_width=640)
        
        print(f"2️⃣  Loading weights dari {PTH_PATH}...")
        # Load weights, map_location biar aman kalau lu convert di PC tanpa GPU
        model.load_state_dict(torch.load(PTH_PATH, map_location=device))
        model.to(device)
        model.eval() # WAJIB EVAL MODE
        
        print("3️⃣  Bikin Dummy Input & Tracing...")
        # Bikin data boongan buat mancing jalur modelnya
        dummy_input = torch.randn(INPUT_SHAPE).to(device)
        
        # MAGIC PROCESS: TRACING
        traced_model = torch.jit.trace(model, dummy_input)
        
        print(f"4️⃣  Saving ke {PT_OUTPUT}...")
        traced_model.save(PT_OUTPUT)
        
        print("\n✅ SUKSES BOS!")
        print(f"   File '{PT_OUTPUT}' siap dideploy (C++, Mobile, dll).")

    except FileNotFoundError:
        print(f"\n❌ GAK KETEMU: File '{PTH_PATH}' mana? Cek path-nya lagi bg.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")