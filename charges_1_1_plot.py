"""
This script generates multiple plots for fitting anions/cations using all 16 formulas and all one properties.
Change variables before `if __name__ == "__main__"` to experiment with different combinations.
PNG file is saved to the `save_dir_path` directory.
"""
import modules as mod
import matplotlib.pyplot as plt
import time
from contextlib import suppress
import os

start = time.perf_counter()
ion_type = "anions"     # edit: anions or cations
file_name = f"csv/{ion_type}.csv"
formula = "Q4R1"
modulename = 'formulas'
property = "dH"
save_dir_path = f"graphs/{property}"

if __name__ == "__main__":
    start = time.perf_counter()
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    formula_func = getattr(mod.formulas, formula)
    df_list = mod.read_data(file_name, property, formula, by_charge=True)
    fig, ax = plt.subplots()
    ax = mod.custom_axis(formula, property, ax)
    for i, each_df in enumerate(df_list):
        with suppress(TypeError):
            prop, fit_data, parameters, SE = mod.fit_parameters(formula, formula_func, property, each_df)
            ax = mod.plot_by_charge(prop, fit_data, ax, ion_type, each_df)
    plt.legend()
    figure_name = f"{ion_type[:3]}_{formula}_{property}"
    fig.savefig(f"{save_dir_path}/{figure_name}", bbox_inches = "tight", dpi = 300, transparent = True)
    end = time.perf_counter()
    print(f'Finished in {round(end-start,2)} seconds')
