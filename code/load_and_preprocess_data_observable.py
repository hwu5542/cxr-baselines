import os
import re
import zipfile
import pandas as pd
from pydicom import dcmread
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import pydicom
from tqdm import tqdm
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

class ObservableDataProcessor:
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        self.current_stage = ""

    def log_progress(self, message):
        logging.info(f"[{self.current_stage}] {message}")

    def preprocess_image(self, pixel_array, target_size=224):
        """Preprocess DICOM image with progress tracking"""
        try:
            img = Image.fromarray(pixel_array)
            img = img.resize((target_size, target_size))
            img = np.array(img, dtype=np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = (img - 0.485) / 0.229
            return img
        except Exception as e:
            self.error_count += 1
            raise e

    def extract_findings_section(self, text):
        """Extract findings section with validation"""
        try:
            findings_start = text.upper().find('FINDINGS:')
            if findings_start == -1:
                return ""
            
            findings_end = text.upper().find('IMPRESSION:')
            if findings_end == -1:
                findings_end = len(text)
            
            return text[findings_start+9:findings_end].strip()
        except Exception as e:
            self.error_count += 1
            raise e

    def build_mapping(self, base_path, reports_folder):
        """Build DICOM-report mapping with progress tracking"""
        self.current_stage = "Building Mapping"
        records = []
        total_patients = len(list(Path(base_path).glob("P*/p*")))
        
        self.log_progress(f"Scanning {total_patients} patient directories...")
        
        for patient_dir in tqdm(Path(base_path).glob("P*/p*"), desc="Patients"):
            patient_id = patient_dir.name[1:]
            
            for study_dir in patient_dir.glob("s[!.]*"):
                study_id = study_dir.name[1:]
                dicom_files = list(study_dir.glob("*.dcm"))
                report_path = Path(reports_folder) / patient_dir.parent.name / patient_dir.name / f"{study_dir.name}.txt"
                
                # print(patient_id, study_id, dicom_files, report_path)
                # return
                if report_path.exists():
                    for dicom_path in dicom_files:
                        records.append({
                            "patient_id": patient_id,
                            "study_id": study_id,
                            "dicom_path": str(dicom_path),
                            "report_path": str(report_path)
                        })
        
        self.log_progress(f"Built mapping for {len(records)} DICOM-report pairs")
        return pd.DataFrame(records)

    def load_data(self, df, image_size=224):
        """Load and preprocess data with progress tracking"""
        self.current_stage = "Loading Data"
        data = []
        self.processed_count = 0
        self.error_count = 0
        
        self.log_progress(f"Processing {len(df)} items...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                dicom = pydicom.dcmread(row["dicom_path"])
                img = self.preprocess_image(dicom.pixel_array, image_size)
                
                with open(row["report_path"]) as f:
                    report = self.extract_findings_section(f.read())
                
                data.append({
                    "image": img,
                    "report": report,
                    "patient_id": row["patient_id"],
                    "study_id": row["study_id"]
                })
                self.processed_count += 1
                
                if self.processed_count % 2000 == 0:
                    self.log_progress(f"Processed {self.processed_count} items")
                    
            except Exception as e:
                self.error_count += 1
                self.log_progress(f"Error processing {row['dicom_path']}: {str(e)}")
                continue
        
        self.log_progress(f"Completed: {self.processed_count} successes, {self.error_count} errors")
        return pd.DataFrame(data)

    def get_train_test_split(self, data_df, test_size=0.2, random_state=42):
        """Create train/test split with logging"""
        self.current_stage = "Splitting Data"
        patient_groups = data_df.groupby('patient_id')
        patients = list(patient_groups.groups.keys())
        
        self.log_progress(f"Splitting {len(patients)} patients...")
        
        train_patients, test_patients = train_test_split(
            patients, test_size=test_size, random_state=random_state)
        
        train_df = pd.concat([patient_groups.get_group(p) for p in tqdm(train_patients, desc="Building train set")])
        test_df = pd.concat([patient_groups.get_group(p) for p in tqdm(test_patients, desc="Building test set")])
        
        self.log_progress(f"Split complete: {len(train_df)} training, {len(test_df)} test items")
        return train_df, test_df

    def save_to_parquet(self, df, output_path):
        self.current_stage = "Saving Data"
        self.log_progress(f"Saving to {output_path}...")
        
        # Convert numpy arrays to bytes
        df = df.copy()
        if 'image' in df.columns:
            df['image'] = df['image'].apply(
                lambda x: x.tobytes() if isinstance(x, np.ndarray) else x
            )
        
        df.to_parquet(output_path)
        self.log_progress(f"Saved {len(df)} items to {output_path}")

    def load_from_parquet(self, file_path):
        self.current_stage = "Loading Data"
        self.log_progress(f"Loading from {file_path}...")
        
        df = pd.read_parquet(file_path)
        
        # Convert bytes back to numpy arrays
        if 'image' in df.columns:
            df['image'] = df['image'].apply(
                lambda x: np.frombuffer(x, dtype=np.float32).reshape(224, 224) 
                if isinstance(x, bytes) else x
            )
        
        self.log_progress(f"Loaded {len(df)} items from {file_path}")
        return df

if __name__ == "__main__":
    processor = ObservableDataProcessor()
    
    # Path configuration
    REPORTS_ZIP = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip"
    REPORTS_FOLDER = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/reports"
    FILE_FOLDER = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/reports/files"
    FILE_PATH = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/files"
    OUTPUT_DIR = "D:/mimic/processed"
    
    try:
        # Step 1: Extract reports
        if not os.path.exists(REPORTS_FOLDER):
            processor.log_progress("Extracting reports...")
            with zipfile.ZipFile(REPORTS_ZIP, 'r') as zip_ref:
                zip_ref.extractall(REPORTS_FOLDER)
        
        # Step 2: Build mapping
        mapping_df = processor.build_mapping(FILE_PATH, FILE_FOLDER)
        
        # Step 3: Process data
        processed_data = processor.load_data(mapping_df)
        
        # Step 4: Split and save
        processor.save_to_parquet(processed_data, os.path.join(OUTPUT_DIR, "processed_data.parquet"))
        train_df, test_df = processor.get_train_test_split(processed_data)
        print('train:', len(train_df))
        print('test: ', len( test_df))
        # processed_data = processor.load_from_parquet(file_path)
        
        processor.log_progress("Data processing completed successfully")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)