import os
import re
import sys
import warnings
import shutil

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, resnet34, wide_resnet50_2

from PIL import Image, ImageOps
from matplotlib import font_manager, rcParams, pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
from scipy.fft import fft2, fftshift
from tkinter import filedialog
import matplotlib.image as mpimg

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

for _font in [r"C:\Windows\Fonts\malgun.ttf",
              r"C:\Windows\Fonts\malgunbd.ttf",
              r"C:\Windows\Fonts\NanumGothic.ttf"]:
    try:
        font_manager.fontManager.addfont(_font)
    except Exception:
        pass

rcParams['font.sans-serif'] = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans', 'Arial', 'Liberation Sans']
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

@dataclass(frozen=True)
class Config:
    # Default settings
    INPUT_DIR: Path = Path()
    REFERENCE_DIR: Optional[Path] = None
    PATIENT_NUM: str = ""
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suffix settings
    SUFFIX_CLASSIFICATION: str = "_classification"
    SUFFIX_CROCS: str = "_crocs"
    SUFFIX_REFERENCE: str = "_reference"

    # Classification settings
    CLASS_MODEL_PATH: Path = Path("code/models/resnet18_classification.pth")
    CLASS_NAMES: Tuple[str, ...] = ('face', 'front', 'left', 'right', 'upper', 'lower')

    # Rotation settings
    ROTATION_MODEL_DIR: Path = Path(".")
    ROTATION_MODEL_PREFIX: str = "code/models/Resnet34_rotation_classification_"
    ROTATION_IMG_SIZE: int = 640
    ROTATION_CLASS_OFFSET: int = 10
    ROTATION_INPUT_CHANNELS: int = 5
    ROTATION_NUM_CLASSES: int = 21

    # Crop settings
    CROP_MODEL_DIR: Path = Path(".")
    CROP_MODEL_PREFIX: str = "code/models/WideResnet50_cropping_"
    CROP_IMG_SIZE: int = 224
    CROP_MOVE_RATIO: float = 0.025
    CROP_ZOOM_RATIO: float = 0.025

    # Thumbnail settings
    THUMBNAIL_SIZE: Tuple[int, int] = (800, 600)

    @property
    def OUTPUT_DIR_CLASSIFICATION(self) -> Path: return self._derive_output_path(self.SUFFIX_CLASSIFICATION)
    @property
    def OUTPUT_DIR_CROCS(self) -> Path: return self._derive_output_path(self.SUFFIX_CROCS)
    @property
    def OUTPUT_DIR_REFERENCE(self) -> Path: return self._derive_output_path(self.SUFFIX_REFERENCE)

    @property
    def has_reference(self) -> bool:
        p = self.REFERENCE_DIR
        return p is not None and p.is_dir() and p.resolve() != self.INPUT_DIR.resolve()

    def _derive_output_path(self, suffix: str) -> Path:
        if not self.INPUT_DIR or not self.INPUT_DIR.name:
            return Path.cwd() / f"output{suffix}"
        return self.INPUT_DIR.parent / f"{self.INPUT_DIR.name}{suffix}"

class AppMode(Enum):
    ROTATION = auto()
    CROP = auto()
    REF_COMPARE = auto()

class UIMode(Enum):
    START = auto()
    MAIN = auto()

@dataclass
class Box:
    cx: float = 0.5
    cy: float = 0.5
    w: float = 1.0
    h: float = 1.0

@dataclass
class ImageInstance:
    path: Path
    category: str
    index: int
    image_data: Image.Image
    thumbnail_data: Image.Image

    initial_angle: float = 0.0
    rotation_angle: float = 0.0
    rotation_history: List[float] = field(default_factory=list, repr=False)

    initial_box: Box = field(default_factory=Box)
    crop_box: Box = field(default_factory=Box)
    crop_history: List[Box] = field(default_factory=list, repr=False)

    @classmethod
    def from_file(cls, file_path: Path, config: Config) -> Optional['ImageInstance']:
        try:
            parts = file_path.stem.split('_')
            category, index = parts[-2], int(parts[-1])
            
            image_data = Image.open(file_path).convert("RGB")
            thumbnail_data = image_data.copy()
            thumbnail_data.thumbnail(config.THUMBNAIL_SIZE)
            
            return cls(
                path=file_path, 
                category=category, 
                index=index, 
                image_data=image_data,
                thumbnail_data=thumbnail_data
            )
        except (IndexError, ValueError):
            warnings.warn(f"Skipping file with unexpected name format: {file_path.name}")
            return None

    def add_rotation(self, angle_delta: float):
        self.rotation_history.append(self.rotation_angle)
        self.rotation_angle += angle_delta

    def undo_rotation(self):
        if self.rotation_history:
            self.rotation_angle = self.rotation_history.pop()

    def add_crop_state(self):
        self.crop_history.append(replace(self.crop_box))

    def undo_crop(self):
        if self.crop_history:
            self.crop_box = self.crop_history.pop()
    
    def get_rotated_thumbnail(self) -> Image.Image:
        return self.thumbnail_data.rotate(self.rotation_angle, resample=Image.BICUBIC, fillcolor='white')
    
    # Version 2
    def get_cropped_thumbnail(self) -> Image.Image:
        rotated_thumb = self.get_rotated_thumbnail()
        
        w, h = rotated_thumb.size
        box = self.crop_box
        left = (box.cx - box.w / 2) * w
        top = (box.cy - box.h / 2) * h
        right = (box.cx + box.w / 2) * w
        bottom = (box.cy + box.h / 2) * h

        return rotated_thumb.crop((left, top, right, bottom))

    def get_final_image(self) -> Image.Image:
        rotated_image = self.image_data.rotate(self.rotation_angle, resample=Image.BICUBIC, fillcolor='white')
        
        w, h = rotated_image.size
        box = self.crop_box
        left = (box.cx - box.w / 2) * w
        top = (box.cy - box.h / 2) * h
        right = (box.cx + box.w / 2) * w
        bottom = (box.cy + box.h / 2) * h

        return rotated_image.crop((left, top, right, bottom))

    def _generate_filename(self, config: Config) -> str:
        angle_int = int(round(self.rotation_angle))
        return f"{config.PATIENT_NUM}_{self.category}_{self.index}{self.path.suffix.lower()}"

    def save_image(self, config: Config) -> Path:
        final_image = self.get_final_image()
        save_path = config.OUTPUT_DIR_CROCS / self._generate_filename(config)
        final_image.save(save_path, quality=95)
        return save_path

@dataclass
class AppState:
    image_data: Dict[str, List[ImageInstance]]
    ref_image_data: Dict[str, List[ImageInstance]]

    current_indices: Dict[str, int]
    ref_current_indices: Dict[str, int]

    selected_class_idx: int = 0
    mode: AppMode = AppMode.ROTATION
    grid_visible: bool = True

class ImageClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _load_model(self) -> nn.Module:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.config.CLASS_NAMES))
        model.load_state_dict(torch.load(self.config.CLASS_MODEL_PATH, map_location=self.config.DEVICE))
        model.to(self.config.DEVICE).eval()
        print("--- Classifying input (and reference) images ---")
        print("✅ Classification model loaded.")
        return model

    def _get_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _classify_single_image(self, image: Image.Image) -> str:
        tensor = self.transform(image).unsqueeze(0).to(self.config.DEVICE)
        with torch.no_grad():
            output = self.model(tensor)
            _, pred = torch.max(output, 1)
        return self.config.CLASS_NAMES[pred.item()]

    def process_directory(self, input_dir: Path, output_dir: Path, progress_cb: Optional[Callable] = None, phase_name: str = "Classification"):
        if not input_dir.exists():
            warnings.warn(f"Input directory '{input_dir}' does not exist.", RuntimeWarning)
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        class_counts = {name: 0 for name in self.config.CLASS_NAMES}
        image_files = sorted([p for p in input_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        total = len(image_files)
        if total == 0 and progress_cb:
            progress_cb(phase_name, 1.0, "처리할 이미지 없음")
            return

        for i, image_path in enumerate(image_files, 1):
            img = Image.open(image_path).convert("RGB")

            predicted_class = self._classify_single_image(img)

            if predicted_class in ['upper', 'lower'] and phase_name == "Classification":
                img = ImageOps.flip(img)

            # Version 2
            if predicted_class in ['upper', 'lower'] and phase_name == "Reference Classification":
                predicted_class = "upper" if predicted_class == "lower" else "lower"

            class_counts[predicted_class] += 1
            new_filename = f"{self.config.PATIENT_NUM}_{predicted_class}_{class_counts[predicted_class]}{image_path.suffix.lower()}"
            save_path = output_dir / new_filename
            img.save(save_path, quality=95)
            
            if progress_cb:
                progress_cb(phase_name, i / total, new_filename)
        
        print(f"📁 {phase_name} output: {output_dir}")

class ResNetRotationClassifier(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.base = resnet34(weights=None)
        self.base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

class WideResNetCropPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        base = wide_resnet50_2(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.LeakyReLU(),
            nn.Linear(512, 128), nn.LeakyReLU(),
            nn.Linear(128, 4), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.reg_head(self.backbone(x))

class RotationPredictor:
    def __init__(self, config: Config):
        self.config = config
        self.models: Dict[str, ResNetRotationClassifier] = self._load_models()

    def _load_models(self) -> Dict[str, ResNetRotationClassifier]:
        models = {}
        for class_name in self.config.CLASS_NAMES:
            model_path = self.config.ROTATION_MODEL_DIR / f"{self.config.ROTATION_MODEL_PREFIX}{class_name}.pt"
            if model_path.exists():
                model = ResNetRotationClassifier(self.config.ROTATION_NUM_CLASSES, self.config.ROTATION_INPUT_CHANNELS)
                model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE))
                model.to(self.config.DEVICE).eval()
                models[class_name] = model
        print(f"✅ Rotation models loaded for: {list(models.keys())}")
        return models

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        def normalize(img_array: np.ndarray) -> np.ndarray:
            min_val, max_val = np.min(img_array), np.max(img_array)
            return (img_array - min_val) / (max_val - min_val + 1e-6)

        gray = cv2.resize(image, (self.config.ROTATION_IMG_SIZE, self.config.ROTATION_IMG_SIZE)).astype(np.float32) / 255.0
        gabor_kernel = cv2.getGaborKernel((31, 31), 4.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor = normalize(cv2.filter2D(gray, cv2.CV_32F, gabor_kernel))
        hp_kernel = np.array([[1, 0, 1], [1, -1, 0], [-9, 10, 3], [6, 0, -1], [0, -1, -1]], dtype=np.float32)
        high_pass = cv2.filter2D(gray, -1, hp_kernel)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        sobel = normalize(np.sqrt(sobel_x**2 + sobel_y**2))
        fft_mag = np.log1p(np.abs(fftshift(fft2(gray))))
        cyclic_spec = normalize(fft_mag)
        stack = np.stack([gray, gabor, high_pass, cyclic_spec, sobel])
        stack = (stack - 0.5) / 0.5
        return torch.tensor(stack, dtype=torch.float32)

    def predict_angle(self, image_instance: ImageInstance) -> float:
        model = self.models.get(image_instance.category)
        if not model:
            return 0.0

        img_np = np.array(image_instance.image_data)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        equalized_gray = cv2.equalizeHist(gray)
        input_tensor = self._preprocess_image(equalized_gray).unsqueeze(0).to(self.config.DEVICE)
        
        with torch.no_grad():
            logits = model(input_tensor)
            pred_class = logits.argmax(dim=1).item()
        
        return float(pred_class - self.config.ROTATION_CLASS_OFFSET)

class CropPredictor:
    def __init__(self, config: Config):
        self.config = config
        self.models: Dict[str, WideResNetCropPredictor] = self._load_models()
        self.transform = transforms.Compose([
            transforms.Resize((self.config.CROP_IMG_SIZE, self.config.CROP_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_models(self) -> Dict[str, WideResNetCropPredictor]:
        models = {}
        for class_name in self.config.CLASS_NAMES:
            model_path = self.config.CROP_MODEL_DIR / f"{self.config.CROP_MODEL_PREFIX}{class_name}.pt"
            if model_path.exists():
                model = WideResNetCropPredictor()
                model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE))
                model.to(self.config.DEVICE).eval()
                models[class_name] = model
        print(f"✅ Cropping models loaded for: {list(models.keys())}")
        return models

    def predict_box(self, image_instance: ImageInstance) -> Tuple[float, float, float, float]:
        model = self.models.get(image_instance.category)
        if not model:
            return 0.5, 0.5, 1.0, 1.0

        input_tensor = self.transform(image_instance.image_data).unsqueeze(0).to(self.config.DEVICE)
        with torch.no_grad():
            pred = model(input_tensor).squeeze(0).cpu().numpy()
        
        cx, cy, w, h = np.clip(pred, 0.0, 1.0)
        return float(cx), float(cy), float(w), float(h)

def natural_keys(text: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def ensure_output_dirs(cfg: Config) -> None:
    cfg.OUTPUT_DIR_CLASSIFICATION.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR_CROCS.mkdir(parents=True, exist_ok=True)
    if cfg.has_reference:
        cfg.OUTPUT_DIR_REFERENCE.mkdir(parents=True, exist_ok=True)

def get_file_list_by_class(directory: Path, class_name: str) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        [f for f in directory.iterdir() if f.is_file() and class_name in f.name],
        key=lambda x: natural_keys(x.name)
    )

def convert_fixed_ratio(w: float, h: float, target_ratio: float, img_w: int, img_h: int) -> Tuple[float, float]:
    image_ratio = img_w / img_h
    target_ratio /= image_ratio

    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = w
        new_h = w / target_ratio
    else:
        new_h = h
        new_w = h * target_ratio

    if new_w > 1.0 or new_h > 1.0:
        scale_factor = 1.0 / max(new_w, new_h)
        new_w *= scale_factor
        new_h *= scale_factor

    return new_w, new_h

def clamp_box_to_bounds(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    clamped_w = float(np.clip(w, 0.1, 1.0))
    clamped_h = float(np.clip(h, 0.1, 1.0))

    half_w = clamped_w / 2.0
    half_h = clamped_h / 2.0
    
    min_cx, max_cx = half_w, 1.0 - half_w
    min_cy, max_cy = half_h, 1.0 - half_h

    clamped_cx = float(np.clip(cx, min_cx, max_cx))
    clamped_cy = float(np.clip(cy, min_cy, max_cy))

    return clamped_cx, clamped_cy, clamped_w, clamped_h

class MainGUI:
    def __init__(self, config: Config, app_state: Optional[AppState] = None):
        self.config = config
        self.state = app_state
        self.mode = UIMode.START

        self._prev_main_mode: AppMode = AppMode.ROTATION
        self._ref_edit_mode: AppMode = AppMode.ROTATION

        self.fig = plt.figure(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title("Image Processing Tool")

        self._build_start_view()

    def _reset_to_start(self):
        self.state = None
        self.mode = UIMode.START
        for ax in self.fig.axes[:]:
            self.fig.delaxes(ax)
        self.fig.suptitle("")
        self.fig.canvas.draw_idle()
        self._build_start_view()

    def _build_start_view(self):
        for ax in self.fig.axes[:]:
            self.fig.delaxes(ax)

        ax_bg = self.fig.add_axes([0, 0, 1, 1])
        ax_bg.axis("off")

        try:
            logo_img = mpimg.imread("assets/logo.png")
            ax_logo = self.fig.add_axes([0.30, 0.72, 0.40, 0.20], anchor='N')
            ax_logo.imshow(logo_img)
            ax_logo.axis("off")
        except FileNotFoundError:
            ax_logo = self.fig.add_axes([0.30, 0.72, 0.40, 0.20], anchor='N')
            ax_logo.axis("off")
            ax_logo.text(0.5, 0.5, "logo.png not found", ha="center", va="center", fontsize=12, color="gray")

        ax_dir_label = self.fig.add_axes([0.2, 0.5, 0.18, 0.05]); ax_dir_label.axis("off")
        ax_dir_label.text(0, 0.5, "대상 폴더:", va="center", fontsize=11)
        ax_dir_btn = self.fig.add_axes([0.4, 0.5, 0.18, 0.05])
        self.btn_browse = Button(ax_dir_btn, "폴더 선택")
        self.btn_browse.on_clicked(self._on_browse_clicked)
        ax_dir_show = self.fig.add_axes([0.6, 0.5, 0.18, 0.05]); ax_dir_show.axis("off")
        self._dir_text = ax_dir_show.text(0, 0.5, "(미선택)", va="center", fontsize=10, color="dimgray")
        self._chosen_dir: Optional[Path] = None

        ax_ref_label = self.fig.add_axes([0.2, 0.4, 0.18, 0.05]); ax_ref_label.axis("off")
        ax_ref_label.text(0, 0.5, "레퍼런스 폴더 (선택):", va="center", fontsize=11)
        ax_ref_btn = self.fig.add_axes([0.4, 0.4, 0.18, 0.05])
        self.btn_browse_ref = Button(ax_ref_btn, "폴더 선택")
        self.btn_browse_ref.on_clicked(self._on_browse_ref_clicked)
        ax_ref_show = self.fig.add_axes([0.6, 0.4, 0.18, 0.05]); ax_ref_show.axis("off")
        self._ref_dir_text = ax_ref_show.text(0, 0.5, "(미선택)", va="center", fontsize=10, color="dimgray")
        self._chosen_ref_dir: Optional[Path] = None

        ax_pt_label = self.fig.add_axes([0.2, 0.3, 0.18, 0.05]); ax_pt_label.axis("off")
        ax_pt_label.text(0, 0.5, "환자번호:", va="center", fontsize=11)
        ax_pt_box = self.fig.add_axes([0.4, 0.3, 0.18, 0.05])
        self.tb_patient = TextBox(ax_pt_box, label="", initial=self.config.PATIENT_NUM, hovercolor="0.95")

        ax_start_btn = self.fig.add_axes([0.4, 0.2, 0.18, 0.05])
        self.btn_start = Button(ax_start_btn, "Start")
        self.btn_start.on_clicked(self._on_start_clicked)
        self.fig.canvas.mpl_connect("key_press_event", self._on_start_keypress)
        self.fig.canvas.draw_idle()
        self._create_progress_ui()

    def _create_progress_ui(self):
        if hasattr(self, "_pbar_ax") and self._pbar_ax in self.fig.axes:
            return
        self._pbar_ax = self.fig.add_axes([0.2, 0.1, 0.58, 0.05])
        self._pbar_ax.set_xlim(0, 1)
        self._pbar_ax.set_ylim(0, 1)
        self._pbar_ax.axis("off")
        self._pbar_bg = Rectangle((0, 0), 1, 1, facecolor=(0.92, 0.92, 0.92), edgecolor="black", linewidth=2, zorder=2)
        self._pbar_fg = Rectangle((0, 0), 0, 1, facecolor=(0.18, 0.52, 0.98), edgecolor="black", linewidth=1, zorder=3)
        self._pbar_ax.add_patch(self._pbar_bg)
        self._pbar_ax.add_patch(self._pbar_fg)
        self._pbar_text = self._pbar_ax.text(0.5, 0.5, "0%", ha="center", va="center", fontsize=10)

    def _progress_update(self, phase: str, ratio: float, message: str = ""):
        if not hasattr(self, "_pbar_fg"): return
        ratio = float(max(0.0, min(1.0, ratio)))
        self._pbar_fg.set_width(ratio)
        self._pbar_text.set_text(f"{phase} {int(ratio * 100)}%{(' — ' + message) if message else ''}")
        canvas = self.fig.canvas
        canvas.draw_idle()
        try:
            canvas.flush_events()
        except Exception:
            pass
        plt.pause(0.001)

    def _progress_reset(self):
        if hasattr(self, "_pbar_fg"):
            self._pbar_fg.set_width(0)
            self._pbar_text.set_text("0%")
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def _on_browse_clicked(self, event):
        initial = str(self.config.INPUT_DIR) if self.config.INPUT_DIR.exists() else str(Path.cwd())
        chosen = filedialog.askdirectory(initialdir=initial, title="분류 대상 폴더 선택")
        if chosen:
            self._chosen_dir = Path(chosen)
            shown = str(self._chosen_dir)
            if len(shown) > 42: shown = "..." + shown[-39:]
            self._dir_text.set_text(shown)
            self._dir_text.set_color("dimgray")
            self.fig.canvas.draw_idle()

    def _on_browse_ref_clicked(self, event):
        initial = (str(self.config.REFERENCE_DIR) if (self.config.REFERENCE_DIR and self.config.REFERENCE_DIR.exists()) else str(Path.cwd()))
        chosen = filedialog.askdirectory(initialdir=initial, title="레퍼런스 폴더 선택")
        if chosen:
            self._chosen_ref_dir = Path(chosen)
            shown = str(self._chosen_ref_dir)
            if len(shown) > 42: shown = "..." + shown[-39:]
            self._ref_dir_text.set_text(shown)
            self._ref_dir_text.set_color("dimgray")
            self.fig.canvas.draw_idle()

    def _on_start_keypress(self, event):
        if event.key == "enter":
            self._on_start_clicked(None)

    def _on_start_clicked(self, event):
        if not self._chosen_dir or not self._chosen_dir.exists():
            self._dir_text.set_text("유효한 폴더를 선택하세요.")
            self._dir_text.set_color("crimson")
            self.fig.canvas.draw_idle()
            return

        new_patient = self.tb_patient.text.strip() or self.config.PATIENT_NUM
        new_ref = (self._chosen_ref_dir if (self._chosen_ref_dir and self._chosen_ref_dir.exists()) else None)
        self.config = replace(self.config, INPUT_DIR=self._chosen_dir, REFERENCE_DIR=new_ref, PATIENT_NUM=new_patient)
        ensure_output_dirs(self.config)

        self._dir_text.set_text("분류 및 초기 예측 중… 잠시만 기다려주세요.")
        self._dir_text.set_color("dimgray")
        self._create_progress_ui()
        self._progress_reset()
        self.fig.canvas.draw_idle()

        try:
            classifier = ImageClassifier(self.config)
            classifier.process_directory(self.config.INPUT_DIR, self.config.OUTPUT_DIR_CLASSIFICATION, self._progress_update, "Classification")
            if self.config.has_reference:
                self._progress_reset()
                classifier.process_directory(self.config.REFERENCE_DIR, self.config.OUTPUT_DIR_REFERENCE, self._progress_update, "Reference Classification")

            self._progress_reset()
            self.state = load_data_and_predict_initials(self.config, progress_cb=self._progress_update)

        except Exception as e:
            self._dir_text.set_text(f"오류 발생: {type(e).__name__} — {e}")
            self._dir_text.set_color("crimson")
            self.fig.canvas.draw_idle()
            return

        self._build_main_view()

    def _build_main_view(self):
        self.mode = UIMode.MAIN
        for ax in self.fig.axes[:]:
            self.fig.delaxes(ax)
            
        self.btn_browse = self.btn_browse_ref = self.tb_patient = self.btn_start = None
        self._dir_text = self._ref_dir_text = self._pbar_ax = None
        
        self._axes_default_pos = None
        self._ref_layout_active = False

        plt.rcParams['keymap.quit'] = plt.rcParams['keymap.quit_all'] = plt.rcParams['keymap.save'] = []

        try: self.fig.set_constrained_layout(False)
        except Exception: pass

        self.axs = self.fig.subplots(2, 3, gridspec_kw={'wspace': 0.2, 'hspace': 0.3})
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.08)
        self._add_logo("assets/logo.png")

        if not hasattr(self, "_axes_default_pos") or self._axes_default_pos is None:
            self._axes_default_pos = [ax.get_position().frozen() for ax in self.axs.flat]
        self._ref_layout_active = False

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._update_display()
        self.fig.canvas.draw_idle()
            
    def _add_logo(self, logo_path: str):
        try:
            logo_img = mpimg.imread(logo_path)
            ax_logo = self.fig.add_axes([0.02, 0.80, 0.25, 0.15], anchor='NW')
            ax_logo.imshow(logo_img)
            ax_logo.axis("off")
            try: ax_logo.set_in_layout(False)
            except Exception: pass
        except FileNotFoundError:
            return

    def _on_key_press(self, event):
        key = event.key.lower()

        if key == 'escape':
            plt.close(self.fig)
            return
        if key == 'ctrl+r':
            self._reset_to_start()
            return
            
        if self.mode != UIMode.MAIN:
            return

        if key == 'tab' and self.config.has_reference:
            if self.state.mode != AppMode.REF_COMPARE:
                self._prev_main_mode = self.state.mode
                self._ref_edit_mode = self._prev_main_mode
                self.state.mode = AppMode.REF_COMPARE
            else:
                self.state.mode = self._prev_main_mode
            self._update_display()
            return

        if self.state.mode == AppMode.REF_COMPARE:
            if key == 'ctrl+a': self._navigate_ref_images(-1)
            elif key == 'ctrl+d': self._navigate_ref_images(+1)

        if key == 'g':
            self.state.grid_visible = not self.state.grid_visible
        elif key in [str(i) for i in range(1, 7)]:
            self.state.selected_class_idx = int(key) - 1
        elif key in ['a', 'd']:
            self._navigate_images(1 if key == 'd' else -1)
        elif key == 'm':
            if self.state.mode == AppMode.REF_COMPARE:
                self._ref_edit_mode = (AppMode.CROP if self._ref_edit_mode == AppMode.ROTATION else AppMode.ROTATION)
            else:
                self.state.mode = (AppMode.CROP if self.state.mode == AppMode.ROTATION else AppMode.ROTATION)

        instance = self._get_current_instance()
        if not instance:
            if key == 'ctrl+s': self._save_all()
            self._update_display()
            return

        active_edit_mode = self._ref_edit_mode if self.state.mode == AppMode.REF_COMPARE else self.state.mode

        if active_edit_mode == AppMode.ROTATION:
            self._handle_rotation_keys(key, instance)
        elif active_edit_mode == AppMode.CROP:
            self._handle_crop_keys(key, instance)

        self._update_display()

    def _handle_rotation_keys(self, key: str, instance: ImageInstance):
        if key in ['q', 'e']:
            instance.add_rotation(1.0 if key == 'q' else -1.0)
        elif key == 'ctrl+z':
            instance.undo_rotation()
        elif key == 'ctrl+s':
            self._save_all()

    def _handle_crop_keys(self, key: str, instance: ImageInstance):
        box = instance.crop_box
        move_x, move_y = box.w * self.config.CROP_MOVE_RATIO, box.h * self.config.CROP_MOVE_RATIO
        zoom_w, zoom_h = box.w * self.config.CROP_ZOOM_RATIO, box.h * self.config.CROP_ZOOM_RATIO
        
        instance.add_crop_state()
        
        if key == 'up': box.cy -= move_y
        elif key == 'down': box.cy += move_y
        elif key == 'left': box.cx -= move_x
        elif key == 'right': box.cx += move_x
        elif key == 'q':
            box.w += zoom_w
            box.h += zoom_h
        elif key == 'e':
            box.w = max(0.1, box.w - zoom_w)
            box.h = max(0.1, box.h - zoom_h)
        elif key == 'ctrl+z':
            instance.undo_crop()
        elif key == 'ctrl+s':
            self._save_all()
        else:
            instance.crop_history.pop()
        
        box.cx, box.cy, box.w, box.h = clamp_box_to_bounds(box.cx, box.cy, box.w, box.h)

    def _navigate_images(self, direction: int):
        class_name = self.config.CLASS_NAMES[self.state.selected_class_idx]
        num_images = len(self.state.image_data[class_name])
        if num_images == 0: return
        current_idx = self.state.current_indices[class_name]
        self.state.current_indices[class_name] = (current_idx + direction) % num_images

    def _navigate_ref_images(self, direction: int):
        class_name = self.config.CLASS_NAMES[self.state.selected_class_idx]
        ref_list = self.state.ref_image_data.get(class_name, [])
        if not ref_list: return
        current_idx = self.state.ref_current_indices.get(class_name, 0)
        self.state.ref_current_indices[class_name] = (current_idx + direction) % len(ref_list)

    def _get_current_instance(self) -> Optional[ImageInstance]:
        class_name = self.config.CLASS_NAMES[self.state.selected_class_idx]
        instances = self.state.image_data[class_name]
        return instances[self.state.current_indices[class_name]] if instances else None

    def _update_display(self):
        self._update_suptitle()

        if self.state.mode == AppMode.REF_COMPARE:
            self._update_display_refcompare()
            return

        if getattr(self, "_ref_layout_active", False):
            for ax, pos in zip(self.axs.flat, self._axes_default_pos):
                ax.set_visible(True)
                ax.set_position(pos)
            self._ref_layout_active = False

        for i, ax in enumerate(self.axs.flat):
            ax.clear()
            class_name = self.config.CLASS_NAMES[i]
            instances = self.state.image_data[class_name]
            if not instances:
                self._draw_placeholder(ax, class_name)
            else:
                current_idx = self.state.current_indices[class_name]
                instance = instances[current_idx]
                self._draw_image_and_info(ax, instance, len(instances), current_idx)
            self._apply_styles(ax, i)

        self.fig.canvas.draw_idle()

    def _update_display_refcompare(self):
        axes = self.axs.flat
        for i, ax in enumerate(axes):
            ax.clear()
            ax.set_visible(i in (0, 1))

        if not getattr(self, "_ref_layout_active", False):
            axes[0].set_position([0.08, 0.15, 0.40, 0.60])
            axes[1].set_position([0.52, 0.15, 0.40, 0.60])
            self._ref_layout_active = True

        left_ax, right_ax = axes[0], axes[1]
        class_name = self.config.CLASS_NAMES[self.state.selected_class_idx]

        ref_list = self.state.ref_image_data.get(class_name, [])
        ref_total, ref_idx = len(ref_list), self.state.ref_current_indices.get(class_name, 0)
        if ref_total == 0:
            self._draw_placeholder(left_ax, f"{class_name} [Reference 0/0]")
        else:
            ref_instance = ref_list[ref_idx]
            left_ax.imshow(ref_instance.get_rotated_thumbnail())
            left_ax.set_title(f"{class_name} [Reference {ref_idx + 1}/{ref_total}]", fontsize=10)
        self._apply_styles(left_ax, self.state.selected_class_idx)

        main_list = self.state.image_data.get(class_name, [])
        main_total, main_idx = len(main_list), self.state.current_indices.get(class_name, 0)
        if main_total == 0:
            self._draw_placeholder(right_ax, f"{class_name} [Main 0/0]")
        else:
            main_instance = main_list[main_idx]

            if self._ref_edit_mode == AppMode.CROP:
                display_image = main_instance.get_cropped_thumbnail()
                right_ax.imshow(display_image)
                box = main_instance.crop_box
                right_ax.set_title(f"{class_name} [Main {main_idx + 1}/{main_total}] | Angle: {main_instance.rotation_angle:+.1f} | Crop W:{box.w:.2f} H:{box.h:.2f}", fontsize=10)
            else:
                display_image = main_instance.get_rotated_thumbnail()
                right_ax.imshow(display_image)
                right_ax.set_title(f"{class_name} [Main {main_idx + 1}/{main_total}] | Angle: {main_instance.rotation_angle:+.1f}°", fontsize=10)
                self._draw_crop_box(right_ax, main_instance.crop_box, display_image.size)

        self._apply_styles(right_ax, self.state.selected_class_idx)
        self.fig.canvas.draw_idle()

    def _draw_placeholder(self, ax, class_name):
        ax.imshow(np.ones(self.config.THUMBNAIL_SIZE[::-1] + (3,), dtype=np.uint8) * 240)
        ax.text(0.5, 0.5, "No Images", ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
        ax.set_title(f"{class_name} [0/0]", fontsize=9)

    def _draw_image_and_info(self, ax, instance, total, current_idx):
        rotated_thumb = instance.get_rotated_thumbnail()
        ax.imshow(rotated_thumb)
        title = (f"{instance.category} [{current_idx + 1}/{total}]\n"
                 f"Angle: {instance.rotation_angle:+.1f}°")
        ax.set_title(title, fontsize=9)
        if self.state.mode == AppMode.CROP:
            self._draw_crop_box(ax, instance.crop_box, rotated_thumb.size)

    def _draw_crop_box(self, ax, box, img_size):
        img_w, img_h = img_size
        abs_w, abs_h = img_w * box.w, img_h * box.h
        abs_cx, abs_cy = img_w * box.cx, img_h * box.cy
        rect = Rectangle(
            (abs_cx - abs_w / 2, abs_cy - abs_h / 2), abs_w, abs_h,
            linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

    def _apply_styles(self, ax, class_idx):
        if self.state.grid_visible:
            ax.grid(True, alpha=0.5, linewidth=1, linestyle='-', color='gray')
            ax.tick_params(labelleft=False, labelbottom=False, length=0)
        else:
            ax.axis("off")
        
        edgecolor = 'black' if self.state.mode == AppMode.REF_COMPARE else ('red' if class_idx == self.state.selected_class_idx else 'black')
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=2, edgecolor=edgecolor, facecolor='none', clip_on=False)
        ax.add_patch(rect)

    def _update_suptitle(self):
        common_help = "Select: 1-6 | Prev/Next: A/D | Ref: Tab | Mode: M | Grid: G | Restart: Ctrl+R | Save: Ctrl+S | Undo: Ctrl+Z | Exit: Esc"

        if self.state.mode == AppMode.REF_COMPARE:
            edit_mode_str = self._ref_edit_mode.name.capitalize()
            if self._ref_edit_mode == AppMode.ROTATION:
                help_text = "Rotate: Q/E | Ref Nav: Ctrl+A/D"
            else:
                help_text = "Move: Arrows | Zoom: Q/E | Ref Nav: Ctrl+A/D"
            
            title = (f"Patient: {self.config.PATIENT_NUM}\n"
                     f"Mode: Reference Compare (Editing: {edit_mode_str})\n"
                     f"{common_help}\n{help_text}")
            self.fig.suptitle(title, fontsize=12, y=0.95)
            return

        mode_str = self.state.mode.name.capitalize()
        if self.state.mode == AppMode.ROTATION:
            help_text = "Rotation: Q/E"
        else:
            help_text = "Move: Arrows(↑↓←→) | Zoom IN/OUT: Q/E"
        
        title = f"Patient: {self.config.PATIENT_NUM}\nMode: {mode_str}\n{common_help}\n{help_text}"
        self.fig.suptitle(title, fontsize=12, y=0.95)

    def _save_all(self):
        print("\n--- Saving all images with applied rotations and cropping ---")
        self.config.OUTPUT_DIR_CROCS.mkdir(exist_ok=True)
        for instances in self.state.image_data.values():
            for instance in instances:
                new_name = instance.save_image(self.config)
                print(f"Saved: {new_name}")
        print(f"📁 Crocs output: {self.config.OUTPUT_DIR_CROCS}")

        if self.config.OUTPUT_DIR_CLASSIFICATION.exists():
            try:
                shutil.rmtree(self.config.OUTPUT_DIR_CLASSIFICATION)
            except Exception as e:
                print(f"⚠️ Failed to delete classification directory: {e}")

        if self.config.has_reference and self.config.OUTPUT_DIR_REFERENCE.exists():
            try:
                shutil.rmtree(self.config.OUTPUT_DIR_REFERENCE)
            except Exception as e:
                print(f"⚠️ Failed to delete reference directory: {e}")

    def run(self):
        plt.show()

def load_data_and_predict_initials(config: Config, progress_cb=None) -> AppState:
    print("\n--- Loading data and predicting initial rotation angle and crop box coordinates ---")
    if progress_cb: progress_cb("Models are loading ... (Rotation)", 0)
    rotation_predictor = RotationPredictor(config)
    if progress_cb: progress_cb("Models are loading ... (Cropping)", 0)
    crop_predictor = CropPredictor(config)

    image_data: Dict[str, List[ImageInstance]] = {name: [] for name in config.CLASS_NAMES}
    ref_image_data: Dict[str, List[ImageInstance]] = {name: [] for name in config.CLASS_NAMES}

    all_lists = {name: get_file_list_by_class(config.OUTPUT_DIR_CLASSIFICATION, name) for name in config.CLASS_NAMES}
    total = sum(len(v) for v in all_lists.values())
    done = 0
    if total == 0 and progress_cb: progress_cb("Rotation & Cropping:", 1.0, "분류 결과가 비어 있음")

    for class_name in config.CLASS_NAMES:
        for file_path in all_lists[class_name]:
            instance = ImageInstance.from_file(file_path, config)
            if instance:
                angle = rotation_predictor.predict_angle(instance)
                instance.initial_angle = instance.rotation_angle = angle
                cx, cy, w, h = crop_predictor.predict_box(instance)
                target_ratio = (3.0/4.0) if instance.category == 'face' else (4.0/3.0)
                img_w, img_h = instance.image_data.size
                w, h = convert_fixed_ratio(w, h, target_ratio, img_w, img_h)
                cx, cy, w, h = clamp_box_to_bounds(cx, cy, w, h)
                instance.initial_box = instance.crop_box = Box(cx, cy, w, h)
                image_data[class_name].append(instance)
            done += 1
            if progress_cb: progress_cb("Rotation & Cropping:", done / total, file_path.name)

    if config.has_reference:
        ref_lists = {name: get_file_list_by_class(config.OUTPUT_DIR_REFERENCE, name) for name in config.CLASS_NAMES}
        ref_total = sum(len(v) for v in ref_lists.values())
        ref_done = 0
        if ref_total == 0 and progress_cb: progress_cb("Reference Load:", 1.0, "이미지 없음")

        for class_name in config.CLASS_NAMES:
            for file_path in ref_lists[class_name]:
                instance = ImageInstance.from_file(file_path, config)
                if instance:
                    ref_image_data[class_name].append(instance)
                ref_done += 1
                if progress_cb: progress_cb("Reference Load:", ref_done / max(1, ref_total), file_path.name)

    current_indices = {name: 0 for name in config.CLASS_NAMES}
    ref_current_indices = {name: 0 for name in config.CLASS_NAMES}
    first_idx = next((i for i, name in enumerate(config.CLASS_NAMES) if image_data[name]), 0)

    return AppState(
        image_data=image_data,
        current_indices=current_indices,
        selected_class_idx=first_idx,
        ref_image_data=ref_image_data,
        ref_current_indices=ref_current_indices,
    )
    
if __name__ == "__main__":
    config = Config()
    gui = MainGUI(config)
    gui.run()