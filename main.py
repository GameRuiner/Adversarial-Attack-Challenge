from models.model_insightface import load_insightface_ir50_model

model = load_insightface_ir50_model("models_bin/backbone_ir50_ms1m_epoch120.pth", device="cpu")