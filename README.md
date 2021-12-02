# Transition state theory-inspired neural network for estimation of the viscosity of deep eutectic solvents
This is a implentation of our paper "Transition state theory-inspired neural network for estimation of the viscosity of deep eutectic solvents":


## Requirements
- scikit-learn == 0.24.2
- pytorch == 1.9.0+cu111
- numpy == 1.18.5
- lightgbm == 3.2.1
- pandas == 1.2.4

## Model structure
![Network architecture of the TSTiNet model](./outputs/other%20results/Network.jpg)

## How to use
If you want to use our model to predict the viscosity of specified deep eutectic solvents (DES), you can follow these steps:
1. `git clone https://github.com/fate1997/TSTiNet`
2. `cd TSTiNet/prediction`
3. open the "input.xlsx" file and fill the rows.   
(ATTNTION: DO NOT DELETE THE "example" ROW)
4. `python predict.py`


If you want to train a new model, you can just run the code:

`cd model && python TSTiNet-mixed.py`

## Cite
If you use TSTiNet in your research, please cite:
```
@article{...}
```
