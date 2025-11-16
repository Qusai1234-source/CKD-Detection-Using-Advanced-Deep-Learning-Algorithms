import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List,Dict
from pathlib import Path
import math
import cv2
from loaders import ImageLoader
import sys
class EDA:

    def __init__(self,meta_data:pd.DataFrame,samples:Dict[str,List[np.array]],out_dir="artifacts"):

        self.meta_data=meta_data
        self.samples=samples
        self.out_dir=Path(out_dir)
        self.out_dir.mkdir(parents=True,exist_ok=True)

    def class_counts(self):

        if "file_path" in self.meta_data.columns:
            fp_col="file_path"
        elif "filepath" in self.meta_data.columns:
            fp_col="filepath"
        count=self.meta_data.groupby('class')[fp_col].count().sort_values(ascending=False)
        count.to_csv(self.out_dir/"class_counts.csv")

        fig,ax=plt.subplots(figsize=(8,4))
        sns.barplot(x=count.index,y=count.values,ax=ax)
        ax.set_title("class Distribution")
        ax.set_xlabel("class")
        ax.set_ylabel("counts")
        plt.tight_layout()
        fig.savefig(self.out_dir / "class_counts.png")
        plt.close(fig)
    
    def image_size_distribution(self):

        df=self.meta_data.dropna(subset=["width","height"])
        fig,ax=plt.subplots(1,2,figsize=(12,4))

        sns.histplot(df["height"],bins=40,ax=ax[0])
        ax[0].set_title("Image Height Distribution")

        sns.histplot(df["width"],bins=40,ax=ax[1])
        ax[1].set_title("Image Width Distribution")
        plt.tight_layout()
        fig.savefig(self.out_dir / "size_distribution.png")
        plt.close(fig)
    
    def intensity_distribution(self):

        df=self.meta_data.dropna(subset=["height","width"])
        fig,ax=plt.subplots(figsize=(8,4))
        sns.boxplot(x="class",y="mean_intensity",data=df,ax=ax)
        ax.set_title("Mean Intensity Distribution by Class")
        plt.tight_layout()
        fig.savefig(self.out_dir / "mean_intensity_by_class.png")
        plt.close(fig)
    
    def show_sample_grid(self,per_class=5):
        for cls,imgs in self.samples.items():
            if(len(imgs)==0):continue
            n=min(per_class,len(imgs))
            fig,ax=plt.subplots(1,n,figsize=(3*n,3))
            for i in range(n):
                ax[i].imshow(imgs[i],cmap='gray')
                ax[i].axis('off')
            fig.suptitle(f"Sample Images from class: {cls}")
            plt.tight_layout()
            out_path = self.out_dir / f"samples_{cls}.png"
            fig.savefig(out_path)   
            plt.close(fig)
    def mean_images_per_class(self):

        target = (128,128)  # H,W
        for cls, imgs in self.samples.items():
            if len(imgs)==0: continue
            arr = []
            for img in imgs:
                img_r = cv2.resize(img, (target[1], target[0]))
                arr.append(img_r.astype(np.float32))
            mean_img = np.mean(arr, axis=0).astype(np.uint8)
            plt.imsave(self.out_dir / f"mean_{cls}.png", mean_img)

if __name__=="__main__":
    example_path = r"C:\projects\CKD-Detection-Using-Advanced-Deep-Learning-Algorithms\Data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"  # you can change to the absolute path if you prefer

    loader = ImageLoader(root_dir=example_path, resize=(128,128), sample_per_class=8, verbose=True)
    try:
        df, samples = loader.load_all()
    except FileNotFoundError as e:
        print("ERROR:", e)
        sys.exit(1)

    eda = EDA(meta_data=df, samples=samples, out_dir="artifacts")
    eda.class_counts()
    eda.image_size_distribution()
    eda.intensity_distribution()
    eda.show_sample_grid(per_class=5)
    eda.mean_images_per_class()