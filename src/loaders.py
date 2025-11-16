# src/loaders.py
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from skimage.color import rgb2gray
import pandas as pd

class ImageLoader:
    """Robust image loader with helpful debug prints and auto-descend behavior."""

    # Use lower-case suffixes (including the dot)
    SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, root_dir: str, resize: Tuple[int,int] = None, sample_per_class: int = 50, verbose: bool = True):
        self.root = Path(root_dir)
        self.resize = resize
        self.sample_per_class = sample_per_class
        self.verbose = verbose

    @staticmethod
    def _read_image(image_path: str):
        """Read image robustly and return RGB uint8 image (H,W,3) or None."""
        try:
            arr = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            # handle grayscale
            if img.ndim == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                # 4 channels -> BGRA, 3 channels -> BGR
                if img.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception:
            return None

    def discover_classes(self) -> Tuple[Path, List[Path]]:
        """
        Determine the dataset root that contains class folders.
        Returns (real_root, list_of_class_dirs).
        Auto-descend if the given root contains exactly one folder which itself
        contains class folders with images.
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root}")

        immediate_dirs = [p for p in self.root.iterdir() if p.is_dir()]
        if self.verbose:
            print(f"[discover_classes] immediate subdirs of {self.root}: {[p.name for p in immediate_dirs]}")

        # If no immediate directories, nothing to do
        if not immediate_dirs:
            return self.root, []

        # If root contains exactly one directory, check whether that child contains class dirs with images
        if len(immediate_dirs) == 1:
            child = immediate_dirs[0]
            child_dirs = [p for p in child.iterdir() if p.is_dir()]
            if self.verbose:
                print(f"[discover_classes] single child found: {child.name}, child subdirs: {[p.name for p in child_dirs]}")
            # check whether child_dirs have image files
            found_images = False
            for d in child_dirs:
                for f in d.iterdir():
                    if f.is_file() and f.suffix.lower() in self.SUPPORTED_SUFFIXES:
                        found_images = True
                        break
                if found_images:
                    break
            if found_images:
                if self.verbose:
                    print(f"[discover_classes] auto-descend into {child} as dataset root.")
                return child, child_dirs

        # Otherwise use immediate_dirs as class folders (may still be valid)
        return self.root, immediate_dirs

    def load_all(self) -> Tuple[pd.DataFrame, Dict[str, List[np.ndarray]]]:
        """
        Walk class folders, read images, compute simple metadata, and store sample images.
        Returns (metadata_df, samples_dict).
        """
        real_root, class_dirs = self.discover_classes()
        if not class_dirs:
            if self.verbose:
                print(f"[load_all] No class directories found under {real_root}.")
            return pd.DataFrame(), {}

        if self.verbose:
            print(f"[load_all] Using dataset root: {real_root}")
            print(f"[load_all] Class folders: {[p.name for p in class_dirs]}")

        rows = []
        samples: Dict[str, List[np.ndarray]] = {p.name: [] for p in class_dirs}

        for cls_dir in class_dirs:
            cls_name = cls_dir.name
            # get all files in this class directory whose suffix matches (case-insensitive)
            files = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in self.SUPPORTED_SUFFIXES]
            if self.verbose:
                print(f"[load_all] Class '{cls_name}' -> found {len(files)} files")

            for fp in files:
                try:
                    img = self._read_image(str(fp))
                    if img is None:
                        raise ValueError("Unreadable or corrupted image")

                    h, w = img.shape[:2]
                    gray = rgb2gray(img)   # floats in [0,1]
                    mean_intensity = float(gray.mean())
                    std_intensity = float(gray.std())
                    median_intensity = float(np.median(gray))

                    rows.append({
                        "file_path": str(fp),
                        "class": cls_name,
                        "height": int(h),
                        "width": int(w),
                        "area": int(h * w),
                        "aspect": float(w / h),
                        "mean_intensity": mean_intensity,
                        "std_intensity": std_intensity,
                        "median_intensity": median_intensity
                    })

                    if len(samples[cls_name]) < self.sample_per_class:
                        if self.resize:
                            img_vis = cv2.resize(img, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
                        else:
                            img_vis = img.copy()
                        samples[cls_name].append(img_vis)

                except Exception as e:
                    rows.append({
                        "file_path": str(fp),
                        "class": cls_name,
                        "error": str(e)
                    })

        df = pd.DataFrame(rows)
        if self.verbose:
            print(f"[load_all] Total rows (including errors): {len(df)}")
            print(f"[load_all] Sample counts per class: {{ {', '.join(f'{k}: {len(v)}' for k,v in samples.items())} }}")
        return df, samples


if __name__ == "__main__":
    # Adjust path to your project layout if needed
    # Option A: point directly to inner dataset folder that contains Cyst/Normal/Stone/Tumor
    # Option B: point to the parent Data/ folder (auto-descend will try to find dataset)
    example_path = r"C:\projects\CKD-Detection-Using-Advanced-Deep-Learning-Algorithms\Data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"  # you can change to the absolute path if you prefer

    loader = ImageLoader(root_dir=example_path, resize=(128,128), sample_per_class=8, verbose=True)
    try:
        df, samples = loader.load_all()
    except FileNotFoundError as e:
        print("ERROR:", e)
        sys.exit(1)

    print("\n=== Result summary ===")
    print("Metadata DataFrame shape:", df.shape)
    if not df.empty:
        print(df.head(10))
    else:
        print("DataFrame is empty (no images detected).")

    print("Samples keys and counts:", {k: len(v) for k, v in samples.items()})
