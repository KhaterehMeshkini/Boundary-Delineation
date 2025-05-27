import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from skimage.morphology import medial_axis, opening, closing, skeletonize, dilation, erosion, square
import rasterio
import filemanager as fm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from PIL import Image
from model_att import DynamicAttentionUNet3D
import random
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import tifffile as tiff
from util import accuracy_check_for_batch, compute_iou, CombinedCELSSIMLoss, postprocess_boundary, thin_boundaries, evaluate_boundary_quality
from os.path import splitext
import time
from collections import defaultdict

start_time = time.time()



# Load config
with open("config.ini", "r") as f:
    config = yaml.safe_load(f)

# Paths
data_folder = config['data_folder']
masks_folder = config['masks_folder']
output_folder = config['output_folder']
model_path = config['model_path']
weights_path = config['weights_path']

# Hyperparameters
in_channels = config['in_channels']
num_classes = config['num_classes']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
learning_rate = config['learning_rate']
analysis_type = config['analysis_type']

log_path = os.path.join(output_folder, "results.txt")
with open(log_path, "w") as log_file:
    log_file.write("Training, Validation, and Test Results\n")
    log_file.write("=" * 50 + "\n")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

# Dataset
class CustomDataset(Dataset):
    def __init__(self, data_folder, masks_folder, transform=None):
        self.file_names = os.listdir(data_folder)
        self.data_folder = data_folder
        self.masks_folder = masks_folder
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data = tiff.imread(os.path.join(self.data_folder, self.file_names[idx]))
        mask = tiff.imread(os.path.join(self.masks_folder, self.file_names[idx]))

        

        if self.transform:
            data, mask= self.transform(data, mask)


        # Compute NDVI and BSI 
        # Adjust band indices based on the band order
        red = data[:, 2, :, :]   # Example: Red band at index 3
        nir = data[:, 6, :, :]   # Example: NIR band at index 7
        blue = data[:, 0, :, :]  # Example: Blue band at index 1
        swir = data[:, 8, :, :] # Example: SWIR band at index 10

        ndvi = (nir - red) / (nir + red + 1e-6)
        bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + 1e-6)

        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        data = (data - mean) / (std + 1e-6)
        data[torch.isnan(data)] = 0
        data[torch.isinf(data)] = 0

        # Add new indices to the spectral dimension
        ndvi = ndvi.unsqueeze(1)
        bsi = bsi.unsqueeze(1)
        data = torch.cat([data, ndvi, bsi], dim=1)  # (T, C+2, H, W)

        #only 10m bands
        data = np.delete(data, [3, 4, 5, 7, 8, 9], axis=1)
        
        if analysis_type == 'temporal':
            data = np.transpose(data,(1,0,2,3)) # for spatio-temporal analysis



        data = torch.tensor(data, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        return data, mask
    
    def apply_augmentation(self, data, mask):
        # Randomly choose an augmentation (rotation or flip)
        augmentation = random.choice([self.rotate, self.flip])

        # Apply the chosen augmentation
        data, mask = augmentation(data, mask)

        return data, mask

    def rotate(self, data, mask):
        
        angle = random.choice([90, 180, 270])

        data = np.rot90(data, k=angle // 90, axes=(2, 3))
        mask = np.rot90(mask, k=angle // 90, axes=(0, 1))


        return data, mask

    def flip(self, data, mask):
        
        
        if random.random() > 0.5:
           data = np.flip(data, axis=2)
           mask = np.flip(mask, axis=0)
        if random.random() > 0.5:
           data = np.flip(data, axis=3)
           mask = np.flip(mask, axis=1)

   
        return data, mask  
    
class CustomTransform:
    def __init__(self, dataset):

        self.dataset = dataset

    def __call__(self, data, mask):
        # Apply data augmentation
        data, mask = full_dataset.apply_augmentation(data, mask)

        return torch.tensor(data.copy(), dtype=torch.float32), torch.tensor(mask.copy(), dtype=torch.long)   
    
# Loader  

full_dataset = CustomDataset(data_folder, masks_folder) 
custom_transform = CustomTransform(full_dataset) 
full_dataset.transform = custom_transform

'''
# Compute class weights
if os.path.exists(weights_path):
    class_weights = torch.load(weights_path)
    
    print("Class Weights:", class_weights)
else:

    counts = torch.zeros(num_classes)
    for _, mask in tqdm(full_dataset):
        for c in range(num_classes):
            counts[c] += (mask == c).sum()
    freq = counts / counts.sum()
    class_weights = 1.0 / (freq + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    torch.save(class_weights, weights_path)
    print("Class Weights:", class_weights)
'''

# Class weights
if analysis_type == 'temporal':
    class_weights = torch.tensor([0.2, 0.9]).to(device) 

else:  
    class_weights = torch.tensor([0.2, 0.9]).to(device)   

print("Class Weights:", class_weights)
# Dataloaders
train_len = int(0.7 * len(full_dataset))
val_len = int(0.15 * len(full_dataset))
test_len = len(full_dataset) - train_len - val_len
train_set, val_set, test_set = torch.utils.data.random_split(full_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Model
model = nn.DataParallel(DynamicAttentionUNet3D(in_channels, num_classes)).to(device)
criterion = CombinedCELSSIMLoss(alpha=0.2, class_weights=class_weights.to(device))
#criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)
#criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

if os.path.exists(model_path):
    model = torch.load(model_path)
    print("Model restored.")
else:
    print("Training model...")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for data, mask in train_loader:
            data, mask = data.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, mask) 
            #prob = torch.softmax(output, dim=1) 
            preds = torch.argmax(output, dim=1) 
           

            acc = accuracy_check_for_batch(mask, preds, data.size()[0])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc

        epoch_msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_acc/len(train_loader):.4f}"
        print(epoch_msg)
        with open(log_path, "a") as log_file:
            log_file.write(epoch_msg + "\n")    
        
        scheduler.step()

    torch.save(model, model_path)

# Validation
model.eval()
val_loss = 0.0
val_acc = 0.0
with torch.no_grad():
    for data, mask in val_loader:
        data, mask = data.to(device), mask.to(device)
        output = model(data)
        loss = criterion(output, mask)
        prob = torch.softmax(output, dim=1) 
        preds = torch.argmax(prob, dim=1)
        acc = accuracy_check_for_batch(mask, preds, data.size(0))
        val_loss += loss.item()
        val_acc += acc

val_msg = f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc/len(val_loader):.4f}"
print(val_msg)
with open(log_path, "a") as log_file:
    log_file.write(val_msg + "\n")


# Test and save predictions

all_preds, all_labels = [], []
boundary_metrics_list = []
test_names = [full_dataset.file_names[i] for i in test_set.indices]
with torch.no_grad():
    for idx, (data, mask) in enumerate(test_loader):
        data, mask = data.to(device), mask.to(device)
        output = model(data)
        _, preds = torch.max(output, 1)
        #all_preds.extend(preds.cpu().numpy().flatten())
        #all_labels.extend(mask.cpu().numpy().flatten())
        for i in range(preds.size(0)):
            pred_np = preds[i].cpu().numpy()
            label_np = mask[i].cpu().numpy()

            # Evaluate spatial boundary metrics

            metrics = evaluate_boundary_quality(pred_np, label_np, pixel_size=10)
            boundary_metrics_list.append(metrics)

            processed_pred = postprocess_boundary(pred_np, complete=True, thickness=2, connect_thresh=15)
            #processed_pred = thin_boundaries(pred_np)

            

            # Save processed prediction as png
            base_name = splitext(test_names[idx * test_loader.batch_size + i])[0]


            # Save processed prediction as Geotiff                    

            input_file_path = os.path.join(masks_folder, test_names[idx * test_loader.batch_size + i])
            out_tif_path_pred = os.path.join(output_folder, f"{base_name}_pred.tif")
            out_tif_path_label = os.path.join(output_folder, f"{base_name}_label.tif")
            _, geotransfrom, projection = fm.readGeoTIFF(input_file_path, metadata=True)
            fm.writeGeoTIFF(out_tif_path_pred, processed_pred, geotransfrom, projection)
            fm.writeGeoTIFF(out_tif_path_label, label_np, geotransfrom, projection)
            Image.fromarray((processed_pred * 255).astype('uint8')).save(os.path.join(output_folder, f"{base_name}_pred.png"))
            Image.fromarray((label_np * 255).astype('uint8')).save(os.path.join(output_folder, f"{base_name}_label.png"))

        all_preds.extend(processed_pred.flatten())
        all_labels.extend(label_np.flatten())    

        # for i in range(preds.size(0)):
        #     base_name = splitext(test_names[idx * test_loader.batch_size + i])[0]
        #     Image.fromarray((preds[i].cpu().numpy() * 255).astype('uint8')).save(os.path.join(output_folder, f"{base_name}_pred.png"))
        #     Image.fromarray((mask[i].cpu().numpy() * 255).astype('uint8')).save(os.path.join(output_folder, f"{base_name}_label.png"))



# Measure the execution time
end_time = time.time()
# Convert execution time to minutes
execution_time_minutes = (end_time - start_time) / 60

# Count the number of parameters
total_params = sum(p.numel() for p in model.parameters())

test_accuracy = accuracy_score(all_labels, all_preds)
test_precision = precision_score(all_labels, all_preds, zero_division=0)
test_recall = recall_score(all_labels, all_preds, zero_division=0)
test_f1 = f1_score(all_labels, all_preds, zero_division=0)
test_kappa = cohen_kappa_score(all_labels, all_preds)
test_iou = compute_iou(all_labels, all_preds)
test_cm = confusion_matrix(all_labels, all_preds)

aggregated = defaultdict(list)
for m in boundary_metrics_list:
    for k, v in m.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                aggregated[f"unit_{subk}"].append(subv)
        else:
            aggregated[k].append(v)

avg_distance = np.nanmean(aggregated["avg_distance_m"])
pct_within_10m = np.nanmean(aggregated["pct_within_10m"])
pct_within_20m = np.nanmean(aggregated["pct_within_20m"]) 
#total_ref_objects = aggregated["total_ref_objects"]
#matched_pairs = np.nanmean(aggregated["matched_pairs"])
#num_objetcs_matches = aggregated["num_matched_objects"]
         

test_msg = (
    "\nTest Results\n"
    f"Accuracy: {test_accuracy:.4f}\n"
    f"Precision: {test_precision:.4f}\n"
    f"Recall: {test_recall:.4f}\n"
    f"F1 Score: {test_f1:.4f}\n"
    f"Kappa Index: {test_kappa:.4f}\n"
    f"IoU: {test_iou:.4f}\n"
    f"Confusion Matrix:\n{test_cm}\n"
    "\nSpatial Boundary Evaluation\n"
    f"Average distance to ref (m): \n{avg_distance:.2f}\n"
    f"% boundary within 10m: {pct_within_10m:.2f}\n"
    f"% boundary within 20m: {pct_within_20m:.2f}\n"
    "\nUnit Matching\n"
    #f"Total reference objetcs: {total_ref_objects:.2f}\n"
    #f"Totalmatched objetcs: {num_objetcs_matches:.2f}\n"
    #f"Matched reference objects: {num_objetcs_matches}/{total_ref_objects}"
    "\nComputations\n"
    f"Execution time:\n{execution_time_minutes:.4f}\n"
    f"Total number of parameters:\n{total_params}\n"

)

print(test_msg)

with open(log_path, "a") as log_file:
    log_file.write(test_msg)




