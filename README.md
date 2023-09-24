# FedBN: FL_GNN

## Usage
### Setup
**pip**

See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```
**conda**

We recommend using conda to quick setup the environment. Please use the following commands.
```bash
conda env create -f environment.yaml
conda activate fedbn
```
### Dataset
**Benchmark(Digits)**
-Dataset are already provided

### Train
Federated Learning

Please using following commands to train a model with federated learning strategy.
- **--mode** specify federated learning strategy, option: fedavg | fedprox | fedbn 
```bash
cd federated
# benchmark experiment
python fed_digits.py --mode fedbn

SingleSet

Please using following commands to train a model using singleset data.
- **--data** specify the single dataset
```bash
cd singleset 
# benchmark experiment, --data option: Caltech|Yale|
python single_digits.py --data Caltech

### Test

```bash
cd federated
# benchmark experiment
python fed_digits.py --mode fedbn --test

