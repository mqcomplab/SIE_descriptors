# This script generates plot for fitting all charges using one formula and one property.
import modules as mod
import matplotlib.pyplot as plt
import time
import os

start = time.perf_counter()
file_name = f"csv/ions.csv"
formula = "Q7R2"
property = "dH"

if __name__ == "__main__":
    formula_func = getattr(mod.formulas, formula)
    df = mod.read_data(file_name, property, formula, by_charge=False)
    prop, fit_data, parameters, SE = mod.fit_parameters(formula, formula_func, property, df)
    fig, ax = plt.subplots()
    ax = mod.custom_axis(formula, property, ax)
    ax = mod.plot_all_charges(fit_data, prop, ax, df)
    dir_name = f"graphs/{property}"
    figure_name = f"{formula}_{property}"
    my_path = os.path.abspath(dir_name)
    fig.savefig(os.path.join(my_path, figure_name), bbox_inches = "tight", dpi = 300, transparent = True)
    print(f"Finished with {formula}: {property}")
end = time.perf_counter()
print(f'Finished in {round(end-start,2)} seconds')
