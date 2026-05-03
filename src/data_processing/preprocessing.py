import os
import cv2
from pathlib import Path
from tqdm import tqdm
import configs.config as cfg 

class HistologyPreprocessor:
    def __init__(self, target_size=(224, 224), apply_clahe=True):
        self.target_size = target_size
        self.use_clahe = apply_clahe
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _apply_clahe_rgb(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    def process_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None: return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.use_clahe:
            img_rgb = self._apply_clahe_rgb(img_rgb)
        img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

def run_preprocessing():
    input_dir = Path(cfg.SPLIT_PATH)         
    output_dir = Path(cfg.PROCESSED_PATH)    
    target_size = cfg.IMG_SIZE              
    
    print(f"Début du prétraitement (Taille: {target_size}, CLAHE: Activé)")
    preprocessor = HistologyPreprocessor(target_size=target_size, apply_clahe=True)

    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_input_dir = input_dir / split
        split_output_dir = output_dir / split
        
        if not split_input_dir.exists(): continue
            
        image_paths = list(split_input_dir.glob('**/*.jpeg')) + list(split_input_dir.glob('**/*.png'))
        print(f"\nTraitement du set '{split}' ({len(image_paths)} images)...")
        
        for img_path in tqdm(image_paths, desc=split):
            rel_path = img_path.relative_to(split_input_dir)
            dest_path = split_output_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            processed_img = preprocessor.process_image(img_path)
            if processed_img is not None:
                cv2.imwrite(str(dest_path), processed_img)

    print("\nPrétraitement terminé !")