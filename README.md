# HMM-Learn

1. Create virtual environment with Python 3.13.2
```
python -m venv .venv
```
2. Activate venv
```
source .venv/Scripts/activate
```
3. Install libraries
```
pip install hmmlearn tiktoken
pip install git+https://github.com/hanson-hschang/Signal-System.git
```
4. Create data using the prepare_data folder
```
```

Create Data:

Beliefnet
Nanogpt
Spectral
Baumwelch



Pipeline
    generate synthetic data/data from text
    Train models, 
        test loss for all models
        Pull A, C, pi for all baumwelch beliefnet
        Eigenvalues for baumwelch, beliefnet
        