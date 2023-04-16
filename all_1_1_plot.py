"""
This script generates a plot for the prediction of specific ion effect property values by fitting all ions to charge transfer model. 

Args:
    file_name (str): the name of the CSV file containing the data
    formula (str): the chemical formula of the ion
    property (str): the thermodynamic property to plot
    save_dir_path (str): the directory to save the plot image

Output:
    A plot of the specified thermodynamic property for the Q7R2 ion, saved as a PNG image in the specified directory.

Usage:
    python all_1_1_plot.py
    
Notes:
    The script uses modules from `modules`.
    Good for experimenting different variable, will produce one plot.
"""

import modules as mod
import matplotlib.pyplot as plt
import time
import os

file_name = f"csv/ions.csv"
formula = "Q7R2"
property = "dG"
save_dir_path = f"graphs/{property}"

if __name__ == "__main__":
    start = time.perf_counter()
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    formula_func = getattr(mod.formulas, formula)
    df = mod.read_data(file_name, property, formula, by_charge=False)
    prop, fit_data, parameters, SE = mod.fit_parameters(formula, formula_func, property, df)
    fig, ax = plt.subplots()
    ax = mod.custom_axis(formula, property, ax)
    ax = mod.plot_all_charges(fit_data, prop, ax, df)
    figure_name = f"{formula}_{property}"
    fig.savefig(f"{save_dir_path}/{figure_name}", bbox_inches = "tight", dpi = 300, transparent = True)
    print(f"Finished with {formula}: {property}")
    end = time.perf_counter()
    print(f'Finished in {round(end-start,2)} seconds')