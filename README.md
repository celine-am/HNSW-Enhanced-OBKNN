# Effeciency Improvement Of Online Bootstrapping KNN anomaly detector 

This repository explores the integration of Approximate Nearest Neighbor(ANN) search into Online Bootstrapping K-Nearest Neighbor (OBKNN) framework to reduce its computational cost.

The original OBKNN approach relies on an exact brute-force nearest neighbor search across multiple data chunks of high-simensional sepctral data. We therefore investgate an HNSW-based replacement of the exact search aiming to improve computational efficiency while maintaining competitive anomaly detection performance.




You can find the original code of OBKNN and the datasets here:
* **Original OBKNN Code:** (https://github.com/nirojasva/spectral-benchmark.git)
* **Original Raw Data BV3:** (https://drive.uca.fr/d/70aec2976f0e45438eb7/)



## Installation

- Step 1: System-Wide Prerequisites
Before installing the Python packages, please ensure you have the following system-level tools installed:

    - Python 3.11.2

    - C++ Build Tools: Required to compile dependencies in dSalmon.

        - On Ubuntu: sudo apt-get install build-essential

        - On Windows: Install "C++ build tools".



    - Java (JDK): Required to run the capymoa package (ideally idk 17).

- Step 2: Virtual Environment Creation


```bash
python3.11 -m venv env_analysis
source env_analysis/bin/activate

python -m pip install --upgrade pip "setuptools<68.0.0" wheel

pip install "numpy<2.0" Cython  

pip install dSalmon --no-build-isolation

pip install vus==0.0.6 autorank==1.3.0 capymoa==0.9.0 openpyxl==3.1.5

grep -iv '^dSalmon' requirements.txt > /tmp/req_no_dsalmon.txt
pip install -r /tmp/req_no_dsalmon.txt

# install jupyter and register the environment
pip install jupyter ipykernel
python -m ipykernel install --user --name=env_analysis --display-name "Python (env_analysis)"

```



# How to run HNSW-enhanced OBKNN
### Parameters

- chunk_size: size of the chunks (default: 240)
- ensemble_size: size of the ensemble of chunks (default: 240)
- dmetric: distance metric used to compute differences among instances is euclidean.


### Scripts

cd ~/HNSW-ENHANCED-OBKNN

source env_analysis/bin/activate
nohup python code/model_hnsw_OBKNN.py

### Results

The results will be generated directly in a csv format and stored in the \improvement_assessment folder.





## Datasets description (/datasets)
- 
- The last column in each dataset file refers to the anomaly label (1: anomaly, 0:normal).
- The first colum in each dataset file correspond to the timestamp of the recorded spectral instances.
- The rest of columns in each dataset are associated with different wavelenths of the spectral instances.



