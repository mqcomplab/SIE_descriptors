"""
This script generates multiple statistal analysis tables for fitting anions using all 16 formulas and one properties.
Change variables before `if __name__ == "__main__"` to experiment with different combinations.
Output is a LaTex formatted table, which you can directly copy to LaTex editor.
"""

import modules as mod
from tabulate import tabulate
import pandas as pd
from contextlib import suppress
import time
import os

ion_type = "anions"
file_name = f"csv/{ion_type}.csv"
formula_list = ["Q1R1", "Q1R2", "Q2R1", "Q2R2", "Q3R1","Q3R2", "Q4R1", "Q4R2", "Q5R1", "Q5R2", "Q6R1","Q6R2", "Q7R1", "Q7R2", "Q8R1", "Q8R2"]
property = "dH"
charge = -1
save_dir_path = "tables"

if __name__ == "__main__":
    start = time.perf_counter()
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    stats_list = []
    aicc_list = []
    for each, formula in enumerate(formula_list):
        formula_func = getattr(mod.formulas, formula)
        df = mod.read_data(file_name, property, formula, by_charge=False)
        df = df[df.Charge == charge]
        with suppress(TypeError, ZeroDivisionError):
            prop, fit_data, parameters, SE = mod.fit_parameters(formula, formula_func, property, df)
            mean_abs_err, rmse, loocv, aicc = mod.get_errors(prop, fit_data, parameters)
            aicc_list.append(aicc)
            stats_df = mod.print_stats(formula, parameters, SE, mean_abs_err, rmse, loocv, aicc)
            stats_list.append(stats_df)
    stats_result = pd.concat(stats_list, axis=1)
    stats_result = mod.replace_aicc(aicc_list, stats_result)
    stats_result = stats_result.fillna("-")
    with open(f"{save_dir_path}/{ion_type[:3]}{charge}_{property}.txt", "w", encoding="utf-8") as f:
        f.write(tabulate(stats_result, headers="keys", stralign="center", tablefmt='fancy_grid'))
    end = time.perf_counter()
    print(f'Finished in {round(end-start,2)} seconds')