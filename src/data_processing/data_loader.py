import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import configs.config as cfg

def get_dataloaders():
    data_dir = cfg.PROCESSED_PATH
    batch_size = cfg.BATCH_SIZE
    img_size = cfg.IMG_SIZE
    
    allowed_classes = cfg.TASK_CLASSES[cfg.TASK]

    transform_list = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip()] + transform_list)
    test_transform = transforms.Compose(transform_list)

    def create_filtered_dataset(split, transform):
        path = os.path.join(data_dir, split)
        full_dataset = datasets.ImageFolder(root=path, transform=transform)
        
        indices = [i for i, (_, label_idx) in enumerate(full_dataset.samples) 
                   if full_dataset.classes[label_idx] in allowed_classes]
        
        return Subset(full_dataset, indices), full_dataset.classes

    train_set, _ = create_filtered_dataset('train', train_transform)
    val_set, _ = create_filtered_dataset('val', test_transform)
    test_set, _ = create_filtered_dataset('test', test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, allowed_classes