import torch
import torch.nn as nn
from torchvision.models import densenet121
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
import numpy as np

class DenseNetFeatureExtractor:
    def __init__(self, device='cuda:0'):
        # Load pretrained DenseNet121
        self.model = densenet121(pretrained=True)
        
        # Remove classification layers
        self.features = self.model.features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        # Move to device and set eval mode
        self.device = torch.device(device)
        self.features.to(self.device)
        self.features.eval()
        
        # Image preprocessing
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
        ])
    
    def extract_from_array(self, pixel_array):
        """Extract features from numpy array (DICOM pixel data)"""
        # Convert to PIL Image and ensure 3 channels
        img = Image.fromarray(pixel_array).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract spatial features (8x8x1024)
            spatial_features = self.features(img_tensor)  # [1, 1024, 8, 8]
            
            # Global mean pooling to 1024-D
            pooled_features = self.pooling(spatial_features)  # [1, 1024, 1, 1]
            pooled_features = pooled_features.view(-1)  # [1024]
            
        return {
            'spatial': spatial_features.squeeze().cpu().numpy(),  # [1024,8,8]
            'pooled': pooled_features.cpu().numpy()  # [1024]
        }

    def batch_extract_arrays(self, pixel_arrays, batch_size=32):
        """Process multiple numpy arrays efficiently"""
        features = []
        
        for i in tqdm(range(0, len(pixel_arrays), batch_size)):
            batch_arrays = pixel_arrays[i:i+batch_size]
            batch_tensors = torch.stack([
                self.transform(Image.fromarray(arr).convert('RGB')) 
                for arr in batch_arrays
            ]).to(self.device)
            
            with torch.no_grad():
                spatial = self.features(batch_tensors)  # [B,1024,8,8]
                pooled = self.pooling(spatial).squeeze()  # [B,1024]
                
            features.extend([{
                'spatial': s.cpu().numpy(),
                'pooled': p.cpu().numpy()
            } for s, p in zip(spatial, pooled)])
            
        return features

# Example usage
# if __name__ == "__main__":
#     extractor = DenseNetFeatureExtractor()
    
#     # Single image processing
#     sample_features = extractor.extract_features("sample.dcm")
#     print("Spatial features shape:", sample_features['spatial'].shape)  # (1024, 8, 8)
#     print("Pooled features shape:", sample_features['pooled'].shape)  # (1024,)
    
#     # Batch processing (for your parquet data)
#     image_paths = ["path1.dcm", "path2.dcm"]  # Load from your metadata
#     all_features = extractor.batch_extract(image_paths)
    
#     # Save features for later use
#     np.save("spatial_features.npy", np.stack([f['spatial'] for f in all_features]))
#     np.save("pooled_features.npy", np.stack([f['pooled'] for f in all_features]))