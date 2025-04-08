import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

OUTPUT_DIR = "D:/mimic/processed/"

class MIMIC_RE:
    def __init__(self):
        self._cached = {}

    def get(self, pattern, flags=0):
        key = hash((pattern, flags))
        if key not in self._cached:
            self._cached[key] = re.compile(pattern, flags=flags)
        return self._cached[key]

    def sub(self, pattern, repl, string, flags=0):
        return self.get(pattern, flags=flags).sub(repl, string)

    def rm(self, pattern, string, flags=0):
        return self.sub(pattern, '', string)

    def get_id(self, tag, flags=0):
        return self.get(r'\[\*\*.*{:s}.*?\*\*\]'.format(tag), flags=flags)

    def sub_id(self, tag, repl, string, flags=0):
        return self.get_id(tag).sub(repl, string)

class ReportParser:
    def __init__(self):
        self.mimic_re = MIMIC_RE()
        self.section_pattern = self.mimic_re.get(r'^(?P<title>[ \w()]+):', flags=re.MULTILINE)

    def parse_text(self, report_text: str) -> str:
        """Parse a single report text with all original de-identification rules"""
        
        if not isinstance(report_text, str):
            return ""
        report = report_text.lower()
        # Apply all original de-identification rules
        report = self.mimic_re.sub_id(r'(?:location|address|university|country|state|unit number)', 'LOC', report)
        report = self.mimic_re.sub_id(r'(?:year|month|day|date)', 'DATE', report)
        report = self.mimic_re.sub_id(r'(?:hospital)', 'HOSPITAL', report)
        report = self.mimic_re.sub_id(r'(?:identifier|serial number|medical record number|social security number|md number)', 'ID', report)
        report = self.mimic_re.sub_id(r'(?:age)', 'AGE', report)
        report = self.mimic_re.sub_id(r'(?:phone|pager number|contact info|provider number)', 'PHONE', report)
        report = self.mimic_re.sub_id(r'(?:name|initial|dictator|attending)', 'NAME', report)
        report = self.mimic_re.sub_id(r'(?:company)', 'COMPANY', report)
        report = self.mimic_re.sub_id(r'(?:clip number)', 'CLIP_NUM', report)

        report = self.mimic_re.sub((
            r'\[\*\*(?:'
                r'\d{4}'  # 1970
                r'|\d{0,2}[/-]\d{0,2}'  # 01-01
                r'|\d{0,2}[/-]\d{4}'  # 01-1970
                r'|\d{0,2}[/-]\d{0,2}[/-]\d{4}'  # 01-01-1970
                r'|\d{4}[/-]\d{0,2}[/-]\d{0,2}'  # 1970-01-01
            r')\*\*\]'
        ), 'DATE', report)
        report = self.mimic_re.sub(r'\[\*\*.*\*\*\]', 'OTHER', report)
        report = self.mimic_re.sub(r'(?:\d{1,2}:\d{2})', 'TIME', report)

        report = self.mimic_re.rm(r'_{2,}', report, flags=re.MULTILINE)
        report = self.mimic_re.rm(r'the study and the report were reviewed by the staff radiologist.', report)

        # Parse sections
        # print(self.section_pattern.finditer(report))
        # matches = list(self.section_pattern.finditer(report))
        # print(self.mimic_re._cached)
        # parsed_report = {}
        # for match, next_match in zip(matches, matches[1:] + [None]):
        #     title = match.group('title').strip()
        #     start = match.end()
        #     end = next_match.start() if next_match else None
        #     paragraph = report[start:end]
        #     paragraph = self.mimic_re.sub(r'\s{2,}', ' ', paragraph).strip()
        #     parsed_report[title] = paragraph.replace('\n', '\\n')
        # print(parsed_report)

        # Find the findings section
        findings_start = report.find('findings:')
        if findings_start == -1:
            return ""
            
        findings_end = report.find('impression:')
        if findings_end == -1:
            findings_end = len(report)
            
        parsed_report = report[findings_start+9:findings_end].strip()
        parsed_report = self.mimic_re.sub(r'\s{2,}', ' ', parsed_report)
        return parsed_report.replace('\n', ' ') if parsed_report else None
        # return parsed_report

    # def parse_file(self, file_path: str) -> Dict[str, str]:
    #     """Maintain original file parsing functionality"""
    #     with open(file_path, 'r') as f:
    #         return self.parse_text(f.read())

    # def batch_parse(self, reports: pd.Series) -> pd.DataFrame:
    #     """Parse a series of reports from parquet data"""
    #     return reports.apply(self.parse_text).apply(pd.Series)

    def parse_parquet(self, input_path: str, output_path: str):
        """Process parquet file and save parsed reports"""
        df = pd.read_parquet(input_path)

        # Extract just the findings from each report
        df['findings'] = df['report'].apply(self.parse_text)

        # Remove records where findings is None or empty string
        df = df[df['findings'].notna() & (df['findings'] != '')]

        # Drop the original report column
        df = df.drop(columns=['report'])
        
        print(df.columns.to_list())
        print(df['findings'])
        df['findings'].to_csv('report_parser.log', sep='\n', index=False)

        # Save the modified dataframe
        df.to_parquet(output_path)

        # print("check 1")
        # print(df.columns.to_list())
        # parsed_df = self.batch_parse(df['report'])
        # print("check 2")
        # parsed_df['findings'].to_csv('report_parser.log', sep='\n', index=False)
        # df = df.drop(columns=['report']).join(parsed_df)
        # df.to_parquet(output_path)

# Example usage
if __name__ == "__main__":
    parser = ReportParser()
    
    # 1. Parse single report text
    sample_text = "FINDINGS: No pneumothorax. IMPRESSION: Normal study."
    # print(parser.parse_text(sample_text))
    
    # 2. Process parquet file
    parser.parse_parquet(
        input_path=os.path.join(OUTPUT_DIR, "train.parquet"),
        output_path=os.path.join(OUTPUT_DIR, "parsed_train.parquet")
    )

    parser.parse_parquet(
        input_path=os.path.join(OUTPUT_DIR, "test.parquet"),
        output_path=os.path.join(OUTPUT_DIR, "parsed_test.parquet")
    )