# cxr-baselines
Baselines for generating radiology reports on the MIMIC-CXR chest x-ray dataset.

Instruction for Windows 11:<br/>
>&nbsp;&nbsp;1. Create mimic folder in D:\ <br/>
&nbsp;&nbsp;2. Copy wget.exe in the mimic directory<br/>
&nbsp;&nbsp;3. Open powershell and run command:
>
>        wget.exe -r -N -c -np --user wuho --ask-password https://physionet.org/files/mimic-cxr/2.1.0/
>&nbsp;&nbsp;4. Clone the git repo and go to code folder<br/>
&nbsp;&nbsp;5. Run python load_and_preprocess_data_observable.py<br/>
&nbsp;&nbsp;&nbsp;&nbsp;(For testing mode, run python load_and_preprocess_data_observable.py numberOfRecordToRun)<br/>
&nbsp;&nbsp;6.

Use

Load and Preprocess Data:

    python load_and_preprocess_data_observable.py

Evaluation:

Powershell:
    
    .\rebuild.ps1

Bash

    .\rebuild.sh

Test Run

    python3 evaluation.py NumberOfRecordToRun the test
    
Actual Run

    python3 evaluation.py
            
            
