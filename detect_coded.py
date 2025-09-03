import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torch.serialization

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# =============================
# Paths (resolved relative to project root)
# =============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Prefer the nested Model/Model/weights first (matches workspace), then fallback
_weights_candidates = [
    PROJECT_ROOT / "Model" / "Model" / "weights" / "best.pt",
    PROJECT_ROOT / "Model" / "weights" / "best.pt",
]
for _candidate in _weights_candidates:
    if _candidate.exists():
        WEIGHTS = str(_candidate)
        break
else:
    WEIGHTS = str(_weights_candidates[0])  # default (may not exist, handled later)

# Default demo source from Test folder
SOURCE = str((PROJECT_ROOT / "Test" / "world.mp4").resolve())

# Output goes to Results folder (ensure it exists)
OUTPUT_DIR = PROJECT_ROOT / "Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = str((OUTPUT_DIR / "output_world.mp4").resolve())
IMG_SIZE = 640
CONF_THRES = 0.5
IOU_THRES = 0.45
DEVICE = ''  # '' for CPU, or '0' for GPU if available

# =============================
# 🩹 Patch torch.load for PyTorch ≥2.6
# =============================
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

def detect():
    source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE
    set_logging()
    device = select_device(DEVICE)

    half = device.type != 'cpu'

    # Validate weights path and load (PyTorch 2.6+ compatibility)
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    # Force load once to bypass pickling guard issues, then let attempt_load create model
    _ = torch.load(str(weights_path), map_location=device, weights_only=False)
    model = attempt_load(str(weights_path), map_location=device)  # load FP32 model

    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    dataset = LoadImages(source, img_size=imgsz)
    #dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Initialize video writer
    vid_writer = None
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=2)

        # Initialize video writer on first frame
        if vid_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = im0s.shape[:2]
            vid_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (w, h))

        vid_writer.write(im0s)

    if vid_writer:
        vid_writer.release()

    print(f'✅ Done. Results saved to {OUTPUT_VIDEO} ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    detect()
