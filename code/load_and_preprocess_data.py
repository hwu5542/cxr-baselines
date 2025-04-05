import os
import zipfile
import pandas as pd
from pydicom import dcmread
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import pydicom

def preprocess_image(pixel_array, target_size=224):
    """Preprocess DICOM image to match model input requirements"""
    # Convert to Hounsfield Units if CT (not needed for MIMIC-CXR)
    if hasattr(pixel_array, 'RescaleSlope') and hasattr(pixel_array, 'RescaleIntercept'):
        pixel_array = pixel_array.RescaleSlope * pixel_array.pixel_array + pixel_array.RescaleIntercept
    
    # Convert to PIL Image for processing
    img = Image.fromarray(pixel_array)
    
    # Resize and normalize
    img = img.resize((target_size, target_size))
    img = np.array(img, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
    
    # Standardize with ImageNet stats if needed
    img = (img - 0.485) / 0.229  # Channel-wise normalization
    
    return img

def extract_findings_section(text):
    """Extract FINDINGS section from radiology report"""
    findings_start = text.upper().find('FINDINGS:')
    if findings_start == -1:
        return ""
    
    findings_end = text.upper().find('IMPRESSION:')
    if findings_end == -1:
        findings_end = len(text)
    
    return text[findings_start+9:findings_end].strip()

def build_mapping(base_path, reports_folder):
    """Create mapping between DICOM images and text reports"""
    records = []
    
    for patient_dir in Path(base_path).glob("P*/p*"):
        patient_id = patient_dir.name[1:]  # Remove 'p' prefix
        
        for study_dir in patient_dir.glob("s*"):
            study_id = study_dir.name[1:]  # Remove 's' prefix
            
            # Get DICOM paths
            dicom_files = list(study_dir.glob("*.dcm"))
            
            # Get report path
            report_path = Path(reports_folder) / patient_dir.parent.name / patient_dir.name / f"{study_dir.name}.txt"
            
            if report_path.exists():
                for dicom_path in dicom_files:
                    records.append({
                        "patient_id": patient_id,
                        "study_id": study_id,
                        "dicom_path": str(dicom_path),
                        "report_path": str(report_path)
                    })
    
    return pd.DataFrame(records)

def load_data(df, image_size=224):
    """Load and preprocess DICOM images with reports"""
    data = []
    
    for _, row in df.iterrows():
        try:
            # Load DICOM
            dicom = pydicom.dcmread(row["dicom_path"])
            img = preprocess_image(dicom.pixel_array, image_size)
            
            # Load report
            with open(row["report_path"]) as f:
                report = extract_findings_section(f.read())
                
            data.append({
                "image": img,
                "report": report,
                "patient_id": row["patient_id"],
                "study_id": row["study_id"]
            })
        except Exception as e:
            print(f"Error processing {row['dicom_path']}: {str(e)}")
    
    return pd.DataFrame(data)

def get_train_test_split(data_df, test_size=0.2, random_state=42):
    """Split data into training and testing sets with patient stratification"""
    # Group by patient to ensure no leakage
    patient_groups = data_df.groupby('patient_id')
    patients = list(patient_groups.groups.keys())
    
    # Split patients
    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=random_state)
    
    # Create splits
    train_df = pd.concat([patient_groups.get_group(p) for p in train_patients])
    test_df = pd.concat([patient_groups.get_group(p) for p in test_patients])
    
    return train_df, test_df

# Main execution
if __name__ == "__main__":
    # Path configuration
    REPORTS_ZIP = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip"
    REPORTS_FOLDER = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/reports"
    FILE_PATH = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/files"
    OUTPUT_DIR = "D:/mimic/processed"
    
    # Step 1: Extract reports if needed
    if not os.path.exists(REPORTS_FOLDER):
        with zipfile.ZipFile(REPORTS_ZIP, 'r') as zip_ref:
            zip_ref.extractall(REPORTS_FOLDER)
    
    # Step 2: Build mapping
    mapping_df = build_mapping(FILE_PATH, REPORTS_FOLDER)
    
    # Step 3: Load and preprocess data
    processed_data = load_data(mapping_df)
    
    # # Step 4: Split and save
    # save_to_parquet(train_df, os.path.join(OUTPUT_DIR, "train.parquet"))
    # save_to_parquet(test_df, os.path.join(OUTPUT_DIR, "test.parquet"))
    
    # Save preprocessed data
    processed_data.to_parquet("preprocessed_data.parquet")

    # Load preprocessed data
    # processed_data = pd.read_parquet(file_path)
    # train_df, test_df = get_train_test_split(processed_data)