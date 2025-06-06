# requirements: pip install open_clip_torch torchvision torch
import torch
import torch.nn as nn
import open_clip
from torchvision.models.video import r3d_18, R3D_18_Weights
from ultralytics import YOLO



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
        x = self.transformer(x)       # (B, T, D)
        x = x.mean(dim=1)            # global average pooling over time
        return x

class CrossTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, query, context):
        """
        query: (B, 1, D) - temporal token (video embedding)
        context: (B, S, D) - e.g., 3D R3D tokens
        """
        x = torch.cat([query, context], dim=1)  # (B, 1+S, D)
        x = self.encoder(x)                     # (B, 1+S, D)
        return x[:, 0]                          # return fused query token


class MultiStreamVideoClassifier(nn.Module):
    def __init__(self, num_classes=4, text_model="ViT-B-32", visual_dim=512):
        super().__init__()

        # 2D CLIP Encoder
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            text_model, pretrained='laion2b_s34b_b79k'
        )
        self.tokenizer = open_clip.get_tokenizer(text_model)
        self.clip_dim = self.clip_model.visual.output_dim

        # 3D Encoder (R3D)
        self.r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.r3d.fc = nn.Identity()
        self.r3d_dim = 512  

        # Projection heads (helps for alignment)
        self.clip_proj = nn.Linear(self.clip_dim, visual_dim)
        self.r3d_proj = nn.Linear(self.r3d_dim, visual_dim)
        self.text_proj = nn.Linear(self.clip_model.text_projection.shape[1], visual_dim)

        # Temporal Transformer
        self.temporal_model = TemporalTransformer(input_dim=visual_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim + visual_dim, 512),  # [video_emb | text_emb]
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, frames_2d, clips_3d, prompts):
        """
        frames_2d: (B, T, C, H, W) - full-resolution RGB frames
        clips_3d: (B, S, C, T, H, W) - segments of short video clips for 3D CNN
        prompts: List[str] of length B
        """


        B, T, C, H, W = frames_2d.shape
        S, C3D, T_3D, H3D, W3D = clips_3d.shape[1:]  # unpack inner dims of clips_3d
#

        ### --- 2D CLIP FEATURES ---
        frames_flat = frames_2d.view(B * T, C, H, W)
        with torch.no_grad():
            clip_2d_feats = self.clip_model.encode_image(frames_flat)  # (B*T, D)
        clip_2d_feats = self.clip_proj(clip_2d_feats)  # project to common dim
        clip_2d_feats = clip_2d_feats.view(B, T, -1)   # (B, T, D)

        ### --- 3D R3D FEATURES ---
        clips_3d = clips_3d.view(B * S, C3D, T_3D, H3D, W3D)
        with torch.no_grad():
            clip_3d_feats = self.r3d(clips_3d)         # (B*S, D)
        clip_3d_feats = self.r3d_proj(clip_3d_feats)
        clip_3d_feats = clip_3d_feats.view(B, S, -1)   # (B, S, D)

        ### --- COMBINE 2D + 3D ---
        combined_feats = torch.cat([clip_2d_feats, clip_3d_feats], dim=1)  # (B, T+S, D)

        ### --- TEMPORAL MODELING ---
        video_emb = self.temporal_model(combined_feats)  # (B, D)

        ### --- TEXT ENCODING ---
        text_inputs = self.tokenizer(prompts)
        with torch.no_grad():
            text_emb = self.clip_model.encode_text(text_inputs)
        text_emb = self.text_proj(text_emb)  # (B, D)

        ### --- LATE FUSION & CLASSIFICATION ---
        fused = torch.cat([video_emb, text_emb], dim=1)  # (B, 2D)
        logits = self.classifier(fused)                  # (B, num_classes)
        return logits

class ClipStyleMultiStreamClassifier(nn.Module):
    def __init__(self, class_prompts, text_model="ViT-B-32", visual_dim=512):
        """
        Args:
            class_prompts: List[str], e.g., ["a video of a novice", ..., "a video of an expert"]
        """
        super().__init__()
        self.class_prompts = class_prompts
        self.num_classes = len(class_prompts)

        # Load CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            text_model, pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer(text_model)
        self.clip_dim = self.clip_model.visual.output_dim

        # 3D CNN (R3D)
        self.r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.r3d.fc = nn.Identity()
        self.r3d_dim = 512

        # Projections
        self.clip_proj = nn.Linear(self.clip_dim, visual_dim)
        self.r3d_proj = nn.Linear(self.r3d_dim, visual_dim)
        self.text_proj = nn.Linear(self.clip_model.text_projection.shape[1], visual_dim)

        # Temporal and cross attention
        self.temporal = TemporalTransformer(input_dim=visual_dim)
        self.cross_fusion = CrossTransformer(dim=visual_dim)

        # CLIP-style logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Precompute text embeddings
        with torch.no_grad():
            text_tokens = self.tokenizer(class_prompts)
            text_features = self.clip_model.encode_text(text_tokens)
            self.register_buffer("text_features", self.text_proj(text_features))  # (num_classes, D)

    def forward(self, frames_2d, clips_3d):
        """
        Args:
            frames_2d: (B, T, C, H, W)
            clips_3d: (B, S, C, T, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = frames_2d.shape
        S, C3D, T3D, H3D, W3D = clips_3d.shape[1:]

        # ----- 2D CLIP visual features -----
        frames_flat = frames_2d.view(B * T, C, H, W)
        with torch.no_grad():
            clip_feats = self.clip_model.encode_image(frames_flat)
        clip_feats = self.clip_proj(clip_feats).view(B, T, -1)  # (B, T, D)

        # Temporal Transformer over 2D tokens
        temporal_summary = self.temporal(clip_feats).unsqueeze(1)  # (B, 1, D)

        # ----- 3D CNN visual features -----
        clips_3d = clips_3d.view(B * S, C3D, T3D, H3D, W3D)
        with torch.no_grad():
            r3d_feats = self.r3d(clips_3d)
        r3d_feats = self.r3d_proj(r3d_feats).view(B, S, -1)  # (B, S, D)

        # ----- Cross Transformer Fusion -----
        fused_video_emb = self.cross_fusion(temporal_summary, r3d_feats)  # (B, D)

        # ----- Normalize embeddings -----
        video_emb = fused_video_emb / fused_video_emb.norm(dim=-1, keepdim=True)
        text_emb = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        # ----- CLIP-style Cosine Similarity -----
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * video_emb @ text_emb.T  # (B, num_classes)
        
        # _, predicted_class = logits.max(dim=1) # if i want to get the predicted class
        return logits
