import os
import zipfile
from pathlib import Path

def download_from_kaggle():
    dataset_slug = "andrewmvd/lung-and-colon-cancer-histopathological-images"
    data_dir = Path("data")
    
    if not data_dir.exists():
        data_dir.mkdir()

    print(f"Téléchargement de {dataset_slug}...")
    os.system(f"kaggle datasets download -d {dataset_slug} -p {data_dir}")
    
    zip_path = data_dir / "lung-and-colon-cancer-histopathological-images.zip"
    
    if zip_path.exists():
        print("Extraction des fichiers...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path) 
        print("Dataset prêt dans le dossier /data")
    else:
        print("Échec : vérifiez vos identifiants Kaggle (~/.kaggle/kaggle.json)")

if __name__ == "__main__":
    download_from_kaggle()