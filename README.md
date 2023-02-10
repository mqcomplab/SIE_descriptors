## Quantifying specific ion effects through charge transfer models
### About
SIE_descriptors allows users to predict specific ion effect properties through magnitudes extracted from
conceptual DFT (C-DFT) to approximate the charge and radius of an ion with or without
perturbed descriptors and using different models of charge transfer in solution.

### Usage
```modules/main.py``` contains all the predefined functions to read csv table, do fitting, and custom plots and tables. All scripts in current directory will run execute those functions. 

The naming scheme of files is: ```{a}_{b}_{c}_{d}.py```

- ```a``` all, cat, ani, charge - fitting all ions, cations, anions, or cations/anions respectively.
- ```b``` 1, formula - reading the script for one or all formulas, respectively. 
- ```c``` 1, prop - reading the script for one or all properties, respectively.
- ```d``` plot, table - plot will produce fitted plot or table will produce error analysis and fit values.
