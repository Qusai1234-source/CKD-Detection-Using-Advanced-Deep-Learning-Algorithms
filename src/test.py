from pathlib import Path
from loaders import ImageLoader
DATA_ROOT = r"C:\projects\CKD-Detection-Using-Advanced-Deep-Learning-Algorithms\Data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
loader = ImageLoader(root_dir=DATA_ROOT, resize=(128,128), sample_per_class=2, verbose=False)

# get class folders and list first N files
paths = []
root = Path(DATA_ROOT)
for cls in sorted([p for p in root.iterdir() if p.is_dir()]):
    files = [str(p) for p in cls.iterdir() if p.is_file()]
    print(cls.name, "count:", len(files))
    paths += files[:50]   # take up to first 50 images per class

print("Total candidate paths:", len(paths))

# check readability
readable = []
unreadable = []
for p in paths:
    img = ImageLoader._read_image(p)
    if img is None:
        unreadable.append(p)
    else:
        readable.append(p)

print("readable:", len(readable), "unreadable:", len(unreadable))
if unreadable:
    print("Sample unreadable files:", unreadable[:5])
