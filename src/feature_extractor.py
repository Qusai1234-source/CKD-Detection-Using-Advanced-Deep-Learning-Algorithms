from tensorflow.keras.applications import ResNet50,MobileNetV2,Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from typing import List,Tuple,Optional
import numpy as np
import os,cv2
from pathlib import Path
from loaders import ImageLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import random
class FeatureExtractor:

    SUPPORTED_MODELS={"resnet50","mobilenetv2","xception"}
    
    def __init__(self,model_name:str="ResNet50" ,input_size:Tuple[int,int,int]=None,batch_size:int =32):

        self.model_name=model_name.lower()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_name: {model_name}. Supported models are: {self.SUPPORTED_MODELS}")
        self.batch_size=batch_size
        
        if input_size is None:
            if self.model_name=="resnet50":
                input_size=(224,224,3)
            elif self.model_name=="mobilenetv2":
                input_size=(224,224,3)
            elif self.model_name=="xception":
                input_size=(299,299,3)
        self.input_size=input_size

        if self.model_name=="resnet50":
            base=ResNet50(include_top=False,pooling='avg',input_shape=self.input_size,weights="imagenet")
            preprocess=resnet_preprocess
        elif self.model_name=="mobilenetv2":
            base=MobileNetV2(include_top=False,pooling='avg',input_shape=self.input_size,weights="imagenet")    
            preprocess=mobilenet_preprocess    
        elif self.model_name=="xception":
            base=Xception(include_top=False,pooling='avg',input_shape=self.input_size,weights="imagenet")
            preprocess=xception_preprocess
        
        self.model=base
        self.preprocess=preprocess
    
    def list_models(self)->List[str]:
        return List(self.SUPPORTED_MODELS)
    def prepare_image(self,img:str)->np.ndarray:

        h,w=self.input_size[0],self.input_size[1]
        img_r=cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)
        arr=img_r.astype(np.float32)
        return arr
    
    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
    
    
        try:
            from loaders import ImageLoader as _IL
            read_fn = _IL.read_image
        except Exception:
            read_fn = None

        out_feats = []
        batch_imgs = []
        batch_indices = []  
        readable_count = 0
        unreadable_count = 0
        preds_count = 0

        for idx, p in enumerate(image_paths):
        
            if read_fn is not None:
                img = read_fn(p)
            else:
                
                try:
                    arr = np.fromfile(p, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        img = None
                    else:
                        if img.ndim == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        elif img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    img = None

            if img is None:
            
                placeholder = np.zeros(self.input_size, dtype=np.float32)
                batch_imgs.append(placeholder)
                batch_indices.append(idx)
                unreadable_count += 1
            else:
                arr = self.prepare_image(img)  # resize + to float32
                batch_imgs.append(arr)
                batch_indices.append(idx)
                readable_count += 1

        
            if len(batch_imgs) == self.batch_size:
                batch_arr = np.stack(batch_imgs, axis=0)  # (B,H,W,3)
                batch_arr = self.preprocess(batch_arr)
                feats = self.model.predict(batch_arr, verbose=0)  # (B, D)
                out_feats.append(feats)
                preds_count += feats.shape[0]
                batch_imgs = []
                batch_indices = []

        if batch_imgs:
            batch_arr = np.stack(batch_imgs, axis=0)
            batch_arr = self.preprocess(batch_arr)
            feats = self.model.predict(batch_arr, verbose=0)
            out_feats.append(feats)
            preds_count += feats.shape[0]

        if not out_feats:
        
            print("[extract_batch] No predictions made; returning empty array.")
            return np.zeros((0, self.model.output_shape[-1]), dtype=np.float32)

        all_feats = np.vstack(out_feats)  
        print(f"[extract_batch] requested: {len(image_paths)} images -> readable: {readable_count}, unreadable(placeholders): {unreadable_count}, predicted rows: {all_feats.shape[0]}")
        return all_feats
    
    def extract_with_mobilenet(self, image_paths: List[str]) -> np.ndarray:
        if self.backbone_name != "mobilenetv2":
            raise RuntimeError("FeatureExtractor not configured with MobileNetV2")
        return self.extract_batch(image_paths)


    def extract_with_resnet(self, image_paths: List[str]) -> np.ndarray:
        if self.backbone_name != "resnet50":
            raise RuntimeError("FeatureExtractor not configured with ResNet50")
        return self.extract_batch(image_paths)


    def extract_with_xception(self, image_paths: List[str]) -> np.ndarray:
        if self.backbone_name != "xception":
            raise RuntimeError("FeatureExtractor not configured with Xception")
        return self.extract_batch(image_paths)
    
    def visualize_features(self, features: np.ndarray, labels: Optional[List[str]] = None,method: str = "pca", title: str = "Feature visualization"):
        if features is None or features.size == 0:
            raise ValueError("Empty features provided")

        n_samples, n_feats = features.shape
        if labels is not None and len(labels) != n_samples:
            raise ValueError(f"labels length ({len(labels)}) != number of feature rows ({n_samples})")

        if n_samples < 2:
            plt.figure(figsize=(4,4))
            plt.scatter([0], [0])
            if labels:
                plt.annotate(labels[0], (0,0))
            plt.title(f"Single-sample embedding (no PCA) - {self.backbone_name}")
            plt.show()
            return

        if method == "pca":
            
            reducer = PCA(n_components=2)
            proj = reducer.fit_transform(features)
        elif method == "tsne":
            
            p = min(30, max(5, n_samples // 3))
            reducer = TSNE(n_components=2, perplexity=p, init="pca", random_state=42)
            proj = reducer.fit_transform(features)
        else:
            raise ValueError("method must be 'pca' or 'tsne'")


        plt.figure(figsize=(8,6))
        if labels is None:
            plt.scatter(proj[:,0], proj[:,1], s=8, alpha=0.8)
        else:
            unique = sorted(set(labels))
            cmap = plt.get_cmap("tab10")
            for i,u in enumerate(unique):
                idx = [j for j,lab in enumerate(labels) if lab == u]
                plt.scatter(proj[idx,0], proj[idx,1], s=20, alpha=0.8, label=str(u), color=cmap(i % 10))
            plt.legend()
        plt.title(f"{title} ({self.model_name} {method})")
        plt.xlabel("dim1"); plt.ylabel("dim2")
        plt.tight_layout()
        plt.show()

if __name__=="__main__":

    fe=FeatureExtractor(model_name="resnet50",input_size=(224,224,3),batch_size=16)
    DATA_ROOT = Path(r"C:\projects\CKD-Detection-Using-Advanced-Deep-Learning-Algorithms\Data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")

    classes = ["Cyst", "Normal","Tumor","Stone"]
    EXAMPLE = []

    for cls in classes:
        cls_dir = DATA_ROOT / cls
        files = [p for p in cls_dir.iterdir() if p.suffix.lower() == ".jpg"]
        EXAMPLE.append(str(random.choice(files)))

    print("Selected examples:", EXAMPLE)

    feats = fe.extract_batch(EXAMPLE)
    print("Features:", feats.shape)
    fe.visualize_features(feats, labels=classes, method="pca", title="Example Feature Visualization")

