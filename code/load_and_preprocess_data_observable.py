import os
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
from report_parser import ReportParser
from save_and_load_parquet import SaveAndLoadParquet
from extract_features import DenseNetFeatureExtractor

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
    def __init__(self, extract_features=True, device='cuda:0', test_mode=False):
        self.processed_count = 0
        self.error_count = 0
        self.current_stage = ""
        self.test_mode = test_mode
        self.extract_features = extract_features
        if extract_features:
            self.feature_extractor = DenseNetFeatureExtractor(device=device)

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
        patient_dirs = list(Path(base_path).glob("P*/p*"))

        if self.test_mode:
            patient_dirs = patient_dirs[:5]  # Only process first 5 patients in test mode
            self.log_progress(f"TEST MODE: Processing first 5 patients only")
        
        self.log_progress(f"Scanning {len(patient_dirs)} patient directories...")
        
        for patient_dir in tqdm(patient_dirs, desc="Patients"):
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

                        # Early exit if we've reached the test mode limit
                        if self.test_mode and len(records) >= 20:
                            self.log_progress(f"TEST MODE: Reached 20 items limit")
                            return pd.DataFrame(records)
        
        self.log_progress(f"Built mapping for {len(records)} DICOM-report pairs")
        return pd.DataFrame(records)

    def load_data(self, df, image_size=224):
        """Load and preprocess data with progress tracking"""
        self.current_stage = "Loading Data"
        data = []
        self.processed_count = 0
        self.error_count = 0
        
        # Apply test mode limit
        if self.test_mode:
            df = df.head(20)
            self.log_progress(f"TEST MODE: Processing first 20 items only")

        self.log_progress(f"Processing {len(df)} items...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                dicom = pydicom.dcmread(row["dicom_path"])
                img = self.preprocess_image(dicom.pixel_array, image_size)
                
                with open(row["report_path"]) as f:
                    report = self.extract_findings_section(f.read())
                
                record = {
                    "image": img,
                    "report": report,
                    "patient_id": row["patient_id"],
                    "study_id": row["study_id"]
                }
                
                if self.extract_features:
                    features = self.feature_extractor.extract_from_array(dicom.pixel_array)
                    record.update({
                        # "spatial_features": features['spatial'],
                        "pooled_features": features['pooled']
                    })
                
                data.append(record)
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
                
        sl = SaveAndLoadParquet()
        sl.save_to_parquet(df, output_path)
        self.log_progress(f"Saved {len(df)} items to {output_path}")

if __name__ == "__main__":
    # Regular Run
    # processor = ObservableDataProcessor()

    # Test Mode On
    processor = ObservableDataProcessor(test_mode=True)
    
    # Path configuration
    REPORTS_ZIP = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip"
    REPORTS_FOLDER = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/reports"
    FILE_FOLDER = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/reports/files"
    FILE_PATH = "D:/mimic/physionet.org/files/mimic-cxr/2.1.0/files"
    OUTPUT_DIR = "D:/mimic/processed"
    OUTPUT_DIR2 = "D:/mimic/outputs"
    
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
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        if not os.path.exists(OUTPUT_DIR2):
            os.makedirs(OUTPUT_DIR2)

        train_df, test_df = processor.get_train_test_split(processed_data)
        print('train:', len(train_df))
        print('test: ', len( test_df))
        processor.save_to_parquet(train_df, os.path.join(OUTPUT_DIR, "train.parquet"))
        processor.save_to_parquet(test_df, os.path.join(OUTPUT_DIR, "test.parquet"))
        
        # Step 5:  Parse parquet file
        parser = ReportParser()

        parser.parse_parquet(
            input_path=os.path.join(OUTPUT_DIR, "train.parquet"),
            output_path=os.path.join(OUTPUT_DIR, "parsed_train.parquet")
        )

        parser.parse_parquet(
            input_path=os.path.join(OUTPUT_DIR, "test.parquet"),
            output_path=os.path.join(OUTPUT_DIR, "parsed_test.parquet")
        )

        processor.log_progress("Data processing completed successfully")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)