from ultralytics import YOLO

# ⚠️ 完全に新しい初期重みから学習を開始（パスの混ざりを防止）
model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset",
    epochs=100,
    imgsz=416,
    batch=16,
    patience=20,
    project="runs_new",       # 👈 保存先フォルダ自体を『runs_new』に新規作成して隔離
    name="screw_2class",
    exist_ok=True,           # 上書きを許可
    cache=False              # 👈 キャッシュの使用を完全に無効化！
)