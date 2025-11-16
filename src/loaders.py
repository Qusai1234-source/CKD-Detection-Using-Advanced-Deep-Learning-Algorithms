import os,glob
from pathlib import Path
from typing import List,Dict,Tuple
import numpy as np
import cv2
from skimage.color import rgb2gray
import pandas as pd

class ImageLoader:

    SUPPORTED_EXTS=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

    def __init__(self,root_dir:str,resize:Tuple[int,int]=None,sample_per_class:int=50):
        self.root= Path(root_dir)
        self.rezise=resize
        self.sample_per_class=sample_per_class

        @staticmethod
        def read_image(image_path:str):
            arr=np.fromfile(image_path,dtype=np.uint8)
            img=cv2.imdecode(arr,cv2.IMREAD_COLOR)

            if img is None:
                return None
            if img.ndim==2:
                img=cv2.cvtColor(img,cv2.color_BGR2RGB)
            elif img.shape[2]==4:
                img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
            else:  
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img
        
        def discover_classes(self)->List[str]:
            classes=[p.name for p in self.root.iterdir() if p.is_dir()]
            classes.sort()
            return classes
        
        def load_all(self)->Tuple[pd.DataFrame,Dict[str,List[np.ndarray]]]:

            rows=[]
            samples={}
            classes=self.discover_classes()

            for cls in classes:
                samples[cls]=[]
                pattern_list=[]

                for ext in self.SUPPORTED_EXTS:
                    pattern_list+=list(self.root.joinpath(cls).glob(ext))
                filePaths=[str(p) for p in pattern_list]
                for i,fp in enumerate(filePaths):

                    try:
                        img=self.read_image(fp)
                        if img is None:
                            raise ValueError(f"Failed to read Image {fp}")

                        h,w=img.shp[:2]
                        gray=rgb2gray(img)
                        mean=float(gray.mean())
                        std=float(gray.std())
                        median=float(np.median(gray))

                        rows.append({
                            "file_path":fp,
                            "class":cls,
                            "height":h,"width":w,
                            "mean_intensity":mean,"std_intensity":std,"median_intensity":median
                        })

                        if len(samples[cls])<self.sample_per_class:
                            if self.resize:
                                img_vis=cv2.resize(img,(self.resize[1],self.resize[0]),interpolation=cv2.INTER_AREA)
                            else:
                                img_vis=img.copy()
                            samples[cls].append(img_vis)
                    except Exception as e:
                        rows.append({
                            "file_path":fp,
                            "class":cls, "Error":str(e)})
            df=pd.DataFrame(rows)
            return df,samples
