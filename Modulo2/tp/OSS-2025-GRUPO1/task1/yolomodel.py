import torch
import torch.nn as nn
import open_clip
from torchvision.models.video import r3d_18, R3D_18_Weights
from ultralytics import YOLO,SAM

# --- Temporal Transformer ---
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer(x)  # (B, T, D)
        return x.mean(dim=1)     # pooled (B, D)

# --- Cross-Attention Transformer ---
class CrossTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, query, context):
        x = torch.cat([query, context], dim=1)  # (B, 1+N, D)
        x = self.encoder(x)
        return x[:, 0]  # Return only the query token

# --- Main Model ---
class ClipStyleMultiStreamClassifierYolo(nn.Module):
    def __init__(self, class_prompts, text_model="ViT-B-32", visual_dim=512, yolo_model_path='yolo11n.pt'):
        super().__init__()  
        self.class_prompts = class_prompts
        self.num_classes = len(class_prompts)

        # --- CLIP ---
        self.clip_model, _, _ = open_clip.create_model_and_transforms(text_model, pretrained="laion2b_s34b_b79k")
        self.tokenizer = open_clip.get_tokenizer(text_model)
        self.clip_dim = self.clip_model.visual.output_dim

        # --- R3D ---
        self.r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.r3d.fc = nn.Identity()
        self.r3d_dim = 512

        # --- YOLO ---
        self.yolo_model = YOLO(yolo_model_path).model
        self.yolo_model.eval()
        for p in self.yolo_model.parameters():
            p.requires_grad = False

        # Hook to capture features
        self.yolo_feat = None
        self._register_yolo_hook()

        # Run dummy input to get YOLO output feature shape
        dummy_input = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            _ = self.yolo_model(dummy_input)
        assert self.yolo_feat is not None, "YOLO hook did not capture features"
        self.yolo_dim = self.yolo_feat.shape[1]

        # --- Projection Layers ---
        self.clip_proj = nn.Linear(self.clip_dim, visual_dim)
        self.r3d_proj = nn.Linear(self.r3d_dim, visual_dim)
        self.yolo_proj = nn.Linear(self.yolo_dim, visual_dim)
        self.text_proj = nn.Linear(self.clip_model.text_projection.shape[1], visual_dim)

        # --- Transformers ---
        self.temporal = TemporalTransformer(input_dim=visual_dim)
        self.cross_fusion = CrossTransformer(dim=visual_dim)

        # --- Logit scale ---
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # --- Text features ---
        with torch.no_grad():
            tokens = self.tokenizer(class_prompts)
            text_feats = self.clip_model.encode_text(tokens)
            self.register_buffer("text_features", self.text_proj(text_feats))

    def _register_yolo_hook(self):
        """
        Attach a forward hook to a reliable intermediate YOLO layer.
        For Ultralytics YOLO, using the last backbone layer is typically stable.
        """
        def hook_fn(module, input, output):
            self.yolo_feat = output

        # Find appropriate layer: last backbone layer is often suitable
        # Change index if YOLO variant differs
        target_layer = list(self.yolo_model.model.modules())[2]  # safe depth for YOLOv8/11
        target_layer.register_forward_hook(hook_fn)

    def forward(self, frames_2d, clips_3d, yolo_frames):
        B, T, C, H, W = frames_2d.shape
        S, C3D, T3D, H3D, W3D = clips_3d.shape[1:]

        # --- CLIP ---
        frames_flat = frames_2d.view(B * T, C, H, W)
        with torch.no_grad():
            clip_feats = self.clip_model.encode_image(frames_flat)
        clip_feats = self.clip_proj(clip_feats).view(B, T, -1)
        temporal_summary = self.temporal(clip_feats).unsqueeze(1)  # (B, 1, D)

        # --- R3D ---
        clips_3d = clips_3d.view(B * S, C3D, T3D, H3D, W3D)
        with torch.no_grad():
            r3d_feats = self.r3d(clips_3d)
        r3d_feats = self.r3d_proj(r3d_feats).view(B, S, -1)  # (B, S, D)

        # --- YOLO ---
        yolo_feats = []
        with torch.no_grad():
            for i in range(B):
                self.yolo_feat = None  # Reset hook buffer
                _ = self.yolo_model(yolo_frames[i].unsqueeze(0))  # (1, 3, 640, 640)
                assert self.yolo_feat is not None, "Hook failed during forward pass"
                pooled = self.yolo_feat.mean(dim=[2, 3])  # Global Average Pooling
                yolo_feats.append(pooled)
        yolo_feats = torch.cat(yolo_feats, dim=0)               # (B, C)
        yolo_feats = self.yolo_proj(yolo_feats).unsqueeze(1)    # (B, 1, D)

        # --- Cross Attention ---
        context = torch.cat([r3d_feats, yolo_feats], dim=1)     # (B, S+1, D)
        fused = self.cross_fusion(temporal_summary, context)    # (B, D)

        # --- Logits ---
        video_emb = fused / fused.norm(dim=-1, keepdim=True)
        text_emb = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * video_emb @ text_emb.T  # (B, num_classes)

        return logits

# --- Full Model ---
class ClipStyleMultiStreamClassifierYoloMultilabel(nn.Module):
    def __init__(self, visual_dim=512, yolo_model_path='yolo11n.pt',
                 text_model="ViT-B-32", num_labels=8, num_classes=5, mlp_hidden_dim=256):
        super().__init__()
        self.num_labels = num_labels
        self.num_classes = num_classes

        # --- CLIP ---
        self.clip_model, _, _ = open_clip.create_model_and_transforms(text_model, pretrained="laion2b_s34b_b79k")
        self.clip_dim = self.clip_model.visual.output_dim

        # --- R3D ---
        self.r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.r3d.fc = nn.Identity()
        self.r3d_dim = 512

        # --- YOLO ---
        self.yolo_model = YOLO(yolo_model_path).model
        self.yolo_model.eval()
        for p in self.yolo_model.parameters():
            p.requires_grad = False

        self.yolo_feat = None
        self._register_yolo_hook()

        dummy_input = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            _ = self.yolo_model(dummy_input)
        assert self.yolo_feat is not None, "YOLO hook did not capture features"
        self.yolo_dim = self.yolo_feat.shape[1]

        # --- Projections ---
        self.clip_proj = nn.Linear(self.clip_dim, visual_dim)
        self.r3d_proj = nn.Linear(self.r3d_dim, visual_dim)
        self.yolo_proj = nn.Linear(self.yolo_dim, visual_dim)

        # --- Transformers ---
        self.temporal = TemporalTransformer(input_dim=visual_dim)
        self.cross_fusion = CrossTransformer(dim=visual_dim)

        # --- MLP Head ---
        self.head = nn.Sequential(
            nn.Linear(visual_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_labels * num_classes)
        )

    def _register_yolo_hook(self):
        def hook_fn(module, input, output):
            self.yolo_feat = output

        # Choose a stable YOLO layer (adjust index if needed)
        target_layer = list(self.yolo_model.model.modules())[2]
        target_layer.register_forward_hook(hook_fn)

    def forward(self, frames_2d, clips_3d, yolo_frames):
        B, T, C, H, W = frames_2d.shape
        S, C3D, T3D, H3D, W3D = clips_3d.shape[1:]
        
        # --- CLIP ---
        frames_flat = frames_2d.view(B * T, C, H, W)
        with torch.no_grad():
            clip_feats = self.clip_model.encode_image(frames_flat)
        clip_feats = self.clip_proj(clip_feats).view(B, T, -1)
        temporal_summary = self.temporal(clip_feats).unsqueeze(1)  # (B, 1, D)

        # print("Done with CLIP")
        # --- R3D ---
        clips_3d = clips_3d.view(B * S, C3D, T3D, H3D, W3D)
        with torch.no_grad():
            r3d_feats = self.r3d(clips_3d)
        r3d_feats = self.r3d_proj(r3d_feats).view(B, S, -1)  # (B, S, D)
        # print("Done with R3D")
        # --- YOLO ---
        yolo_feats = []
        with torch.no_grad():
            for i in range(B):
                self.yolo_feat = None
                _ = self.yolo_model(yolo_frames[i].unsqueeze(0))
                assert self.yolo_feat is not None
                pooled = self.yolo_feat.mean(dim=[2, 3])  # GAP
                yolo_feats.append(pooled)
        yolo_feats = torch.cat(yolo_feats, dim=0)  # (B, C)
        yolo_feats = self.yolo_proj(yolo_feats).unsqueeze(1)  # (B, 1, D)
        # print("Done with YOLO")
        # --- Fusion ---
        context = torch.cat([r3d_feats, yolo_feats], dim=1)  # (B, S+1, D)
        fused = self.cross_fusion(temporal_summary, context)  # (B, D)
        # print("Done with Cross Fusion")
        # --- MLP Head ---
        output = self.head(fused)                  # (B, 8 * 5)
        output = output.view(B, self.num_labels, self.num_classes)  # (B, 8, 5)
        # print("Done with MLP")
        return output