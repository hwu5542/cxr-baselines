import os
import zipfile
import pandas as pd
from pydicom import dcmread
from pathlib import Path

# 1. Unpack reports (if not already done)
REPORTS_ZIP = "mimic-cxr-reports.zip"
if not os.path.exists("reports"):
    with zipfile.ZipFile(REPORTS_ZIP, 'r') as zip_ref:
        zip_ref.extractall("reports")

# 2. Create mapping between images and reports
def build_mapping(base_path="files"):
    records = []
    
    # Walk through patient folders
    for patient_dir in Path(base_path).glob("P*/p*"):
        patient_id = patient_dir.name[1:]  # Remove 'p' prefix
        
        # Process each study
        for study_dir in patient_dir.glob("s*"):
            study_id = study_dir.name[1:]  # Remove 's' prefix
            
            # Get DICOM paths
            dicom_files = list(study_dir.glob("*.dcm"))
            
            # Get report path (from unpacked zip)
            report_path = f"reports/{patient_dir.parent.name}/{patient_dir.name}/{study_dir.name}.txt"
            
            if os.path.exists(report_path):
                for dicom_path in dicom_files:
                    records.append({
                        "patient_id": patient_id,
                        "study_id": study_id,
                        "dicom_path": str(dicom_path),
                        "report_path": report_path
                    })
    
    return pd.DataFrame(records)

# 3. Load and preprocess data (modified from original notebook)
def load_data(df, image_size=224):
    data = []
    
    for _, row in df.iterrows():
        try:
            # Load DICOM
            dicom = dcmread(row["dicom_path"])
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

# Helper functions
def preprocess_image(pixel_array, target_size):
    """Resize and normalize DICOM image"""
    # ... (implementation matches original notebook)
    return processed_img

def extract_findings_section(text):
    """Extract FINDINGS section from report"""
    # ... (implementation matches original notebook)
    return findings_text

# Pipeline execution
if __name__ == "__main__":
    # Step 1: Build mapping
    mapping_df = build_mapping()
    
    # Step 2: Filter AP views (as in paper)
    # ... (add filtering logic if needed)
    
    # Step 3: Load and preprocess
    processed_data = load_data(mapping_df)
    
    # Save preprocessed data
    processed_data.to_parquet("preprocessed_data.parquet")