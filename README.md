# cxr-baselines
Baselines for generating radiology reports on the MIMIC-CXR chest x-ray dataset.
Since I ran the program in Windows 11, the instruction is created using windows system.
For linux and mac, please convert the instructions to fit your system accordingly

Instruction for Windows 11:<br/>
1. Create mimic folder in D:\ (change D:\ to other disk in code if necessary) <br/>
2. Copy wget.exe in the mimic directory<br/>
3. Open powershell and run command:
```
wget.exe -r -N -c -np --user <USERNAME> --ask-password https://physionet.org/files/mimic-cxr/2.1.0/
```
4. Update GPU Driver, then install Cuda toolkit, cuDNN and tensorflow <br/>
- https://developer.nvidia.com/cuda-downloads <br/>
- https://developer.nvidia.com/cudnn-downloads <br/>
- https://www.tensorflow.org/install <br/>
5. Clone the git repo, install requirements and go to code folder <br/>
```
git clone https://github.com/hwu5542/cxr-baselines.git
pip install -r requirements.txt
cd cxr-baselines/code
```
6. Run python load_and_preprocess_data_observable.py<br/>
```
python load_and_preprocess_data_observable.py
```
- ( For testing mode, run ```python load_and_preprocess_data_observable.py <number of records to load>```)<br/>
7. Run each baselines one at a time
```
python randomRetrival.py
python nearest_neighbor.py
python ngram.py
python cnn_rnn_bert.py
```
8. Change to cxr-baselines directory, copy the output files into the evaluate folder as input, then run the building script
```
cd ..
Copy-Item -Path "D:\mimic\outputs\*" -Destination ".\input\" -Recurse
.\rebuild.ps1
```  
9. In the container, run the evaluation:
```
python evaluation.py
```
- Run large batch of evaluation could cause container memory to exceed the limit.\
make sure to make the wsl 2 memory limit bigger by changing the .wslconfig accordingly.\
here is my setup:
```
[wsl2]
memory=24GB       # Minimum 4GB recommended for ML workloads
processors=4     # Allocate CPU cores (e.g., 4)
swap=12GB         # Optional: Adjust swap space
localhostForwarding=true
```
10. Final step, exit the container and run the evaluation program
```
exit
python .\code\displayPredictions.py
python .\evaluate\results_aggregator.py
```
11. open the csv in .\evaluate\output to see the results
