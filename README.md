# A Hierarchical Framework for Solving the Constrained Multiple Depot Traveling Salesman Problem
## Setup Environment
If you do not have a cuda or do not want to run the neural tsp solver in the ablation study, a minimal environment can be setup as
```bash
conda create -n cmdtsp python=3.8
conda activate cmdtsp
pip install -r requirements.txt
```
To run the neural tsp solver, you need pytorch >= 1.7.
## Run
### Test
```bash
python main.py
```
will generate all the data in folder *datasets* and run the comparison and ablation study. 