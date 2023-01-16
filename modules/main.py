import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from adjustText import adjust_text
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression
from contextlib import suppress
import modules.formulas

def get_property_col(property):
    """
    dG: Gibbs Free Energy, dH: Enthalpy, viscosity: Viscosity-B coefficients of electrolyte solutions per coordinating water molecule,
    diffusion: Diffusion coefficients of ions in water, gfet: Gibbs free energies of ion transfer from water to methanol,
    dlcst: ΔLCST of pNIPAM-coated silica particles in 1 M electrolyte solutions, sn2: SN2 reaction rate of iodomethane and ionic nucleophiles in methanol,
    activity: Activity of a human rhinovirus, lysosome: Temperature dependence of the cloud point of lysozyme.
    """
    if property == "dG":
        property = 6
    elif property == "dH":
        property = 8
    elif property == "viscosity":
        property = 9
    elif property == "diffusion":
        property = 10
    elif property == "gfet":
        property = 11
    elif property == "dlcst":
        property = 12
    elif property == "sn2":
        property = 13
    elif property == "activity":
        property = 14
    elif property == "lysozyme":
        property = 15
    else:
        raise ValueError("Not a valid property. The options are dG, dH, viscosity, diffusion, gfet, dlcst, sn2, activity, and lysozyme.")   
    return property

def need_homo_1(formula):
    """ Is it a formula that requires E_HOMO-1 values? """
    homo_1_formulas = ["Q3R1","Q3R2","Q6R1","Q6R2"]
    not_homo_1_formulas = ["Q1R1", "Q1R2", "Q2R1", "Q2R2", "Q4R1", "Q4R2", "Q5R1", "Q5R2", "Q7R1", "Q7R2", "Q8R1", "Q8R2"]
    if any(i in formula for i in homo_1_formulas):
        return True
    elif any(i in formula for i in not_homo_1_formulas):
        return False
    else:
        raise ValueError("Not a valid formula; Only valid are Q[1-8]*R[1-2]*.")

def read_data(file_name, property, formula, by_charge):
    """ 
    Load the dataset. Files must be in CSV UTF-8 format to recognize chemical formulas. All rows containing NaN values will be excluded from the new dataframe.
    by_charge = False for generating one dataframe, stats tables, and plotting all charges. by_charge = True for generating a list of dataframes and plotting by charges. 
    """
    property = get_property_col(property)
    if need_homo_1(formula) is True:
        read_data = pd.read_csv(file_name, delimiter=",", usecols=[0, 1, 2, 3, 4, property], index_col=0, encoding='utf-8')
    elif need_homo_1(formula) is False:
        read_data = pd.read_csv(file_name, delimiter=",", usecols=[0, 1, 2, 3, property], index_col=0, encoding='utf-8')
    df = pd.DataFrame(read_data)
    df = df.dropna()
    if by_charge is False:
        return df
    elif by_charge is True:
        df = df.sort_values("Charge")
        names = df.Charge.unique().tolist()
        df_list = []
        for name in range(len(names)):
            df_list.append(df[df.Charge == names[name]])
        return df_list

def fit_parameters(formula, formula_func, property, df):
    if need_homo_1(formula) is True:
        parameters, covariance = curve_fit(formula_func, (df['E_HOMO'], df['E_LUMO'], df['HOMO-1']), df[property], maxfev = 500000000)
        fit_data = formula_func((df['E_HOMO'], df['E_LUMO'], df['HOMO-1']), *parameters)
    elif need_homo_1(formula) is False:
        parameters, covariance = curve_fit(formula_func, (df['E_HOMO'], df['E_LUMO']), df[property], maxfev = 500000000)
        fit_data = formula_func((df['E_HOMO'], df['E_LUMO']), *parameters)
    prop = df[property]
    SE = np.sqrt(np.diag(covariance))
    return prop, fit_data, parameters, SE

def plot_all_charges(prop, fit_data, ax, df):
    x = np.linspace(min(prop), max(prop))
    y = x
    ax.scatter(prop, fit_data, marker='o')
    ax.plot(x, y, dashes=[4, 4])
    texts = []
    for j, txt in enumerate(df.index):
        texts.append(ax.text(prop[j], fit_data[j], txt, ha='center', va='center'))
    adjust_text(texts,ax=ax)
    return ax

def plot_by_charge(prop, fit_data, ax, ion_type, df):
    z = np.polyfit(prop, fit_data, 1)
    p = np.poly1d(z)
    if ion_type == "cations":
        ax.scatter(prop, fit_data, marker='o', label = f"+ {df.iloc[0, 0]} Charge")
    elif ion_type == "anions":
        ax.scatter(prop, fit_data, marker='o', label = f"{df.iloc[0, 0]} Charge")
    else:
        raise ValueError("cations or anions only")
    ax.plot(prop, p(prop), dashes=[4, 4])
    texts = []
    for j, txt in enumerate(df.index):
        texts.append(ax.text(prop[j], fit_data[j], txt, ha='center', va='center'))
    adjust_text(texts,ax=ax)
    return ax

def custom_axis(formula, property, ax):
    if property == "dG":
        ax.title.set_text(formula + ': ' r'$\mathrm{\Delta G}$')
        ax.set_xlabel('Experimental 'r'$\mathrm{\Delta G_{specific(1:1)-TOT}~(kJ/mol)}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{\Delta G_{specific(1:1)-TOT}~(kJ/mol)}$')

    elif property == "dH":
        ax.title.set_text(formula + ': ' r'$\mathrm{\Delta H}$')
        ax.set_xlabel('Experimental 'r'$\mathrm{-\Delta_h H_l^\infty~per~H_2O~(kJ/mol)}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{-\Delta_h H_l^\infty~per~H_2O~(kJ/mol)}$')

    elif property == "viscosity":
        ax.title.set_text(formula + ': Viscosity-B Coefficients')
        ax.set_xlabel('Experimental 'r'$\mathrm{B_{\eta l}~per~H_2O~(dm^3/mol)}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{B_{\eta l}~per~H_2O~(dm^3/mol)}$')

    elif property == "diffusion":
        ax.title.set_text(formula + ': Diffusion Coefficients')
        ax.set_xlabel('Experimental 'r'$\mathrm{D_l^\infty~(10^{-9}m^2/s)}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{D_l^\infty~(10^{-9}m^2/s)}$')

    elif property == "gfet":
        ax.title.set_text(formula + ': GFET')
        ax.set_xlabel('Experimental 'r'$\mathrm{\Delta G^\infty(I^\pm,~W\rightarrow M)~(kJ/mol)~at~25^{\circ}C}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{\Delta G^\infty(I^\pm,~W\rightarrow M)~(kJ/mol)~at~25^{\circ}C}$')

    elif property == "dlcst":
        ax.title.set_text(formula + ': ' r'$\mathrm{\Delta LCST}$')
        ax.set_xlabel('Experimental 'r'$\mathrm{\Delta LCST~at~1M~(^{\circ}C)}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{\Delta LCST~at~1M~(^{\circ}C)}$')

    elif property == "sn2":
        ax.title.set_text(formula + ': ' r'$\mathrm{S_N2~Rates}$')
        ax.set_xlabel('Experimental 'r'$\mathrm{CH_3I+X^-~\log k^M}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{CH_3I+X^-~\log k^M}$')

    elif property == "activity":
        ax.title.set_text(formula + ': Activity of human rhinovirus')
        ax.set_xlabel('Experimental 'r'$\mathrm{ln(\%~HRV-14~Activation)}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{ln(\%~HRV-14~Activation)}$')

    elif property == "lysozyme":
        ax.title.set_text(formula + ': Temperature dependence of cloud point of lysozyme')
        ax.set_xlabel('Experimental 'r'$\mathrm{Lysozyme~c~(^{\circ}C/M)}$')
        ax.set_ylabel('Theoretical 'r'$\mathrm{Lysozyme~c~(^{\circ}C/M)}$')

    else:
        raise ValueError("Not valid. The options are dG, dH, viscosity, diffusion, gfet, dlcst, sn2, activity, and lysozyme.")

    return ax

def get_errors(prop, fit_data, parameters):
    """
    Generate mean absolute error, root-mean square error and Akaike information criterion corrected for small samples.
    LOOCV: Fit multiple linear regression model to dataset and use Leave-one-out cross validation (LOOCV) to evaluating model performance. Output is the RMSE of the LOOCV.
    """
    mean_abs_err = mean_absolute_error(prop, fit_data)
    rmse = np.sqrt(mean_squared_error(prop, fit_data))
    aicc = modules.calculate_aicc(prop.shape[0], rmse ** 2, parameters.shape[0]) 
    X = pd.DataFrame(fit_data)
    Y= pd.DataFrame(prop)
    cv = LeaveOneOut()
    model = LinearRegression()  
    scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    loocv = np.sqrt(np.mean(np.absolute(scores)))
    return mean_abs_err, rmse, loocv, aicc

def calculate_aicc_min(property, file_name, charge):
    formula_list = ["Q1R1", "Q1R2", "Q2R1", "Q2R2", "Q3R1","Q3R2", "Q4R1", "Q4R2", "Q5R1", "Q5R2", "Q6R1","Q6R2", "Q7R1", "Q7R2", "Q8R1", "Q8R2"]
    list = []
    for each, formula in enumerate(formula_list):
        formula_func = getattr(modules.formulas, formula)
        df = read_data(file_name, property, formula, by_charge=False)
        df = df[df.Charge == charge]
        with suppress(TypeError, ZeroDivisionError):
            prop, fit_data, parameters, SE = fit_parameters(formula, formula_func, property, df)
            rmse = np.sqrt(mean_squared_error(prop, fit_data))
            aicc = modules.calculate_aicc(prop.shape[0], rmse ** 2, parameters.shape[0])
            list.append(aicc)
    if len(list) == 0:
        return np.nan
    else:
        return np.min(list)
    
def replace_aicc(aicc_list, stats_result):
    if len(aicc_list) == 0:
        aicc_min = np.nan
    else:
        aicc_min = np.min(aicc_list)
    daicc_list = list(map(lambda v: v - aicc_min, aicc_list))
    stats_result.iloc[3]= ['%.2f' % elem for elem in daicc_list]
    return stats_result

def print_stats(formula, parameters, SE, mean_abs_err, rmse, loocv, aicc):
    if formula == "Q1R1":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1])]}
        
    elif formula == "Q1R2":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "α"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2])]}

    elif formula == "Q2R1":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2])]}

    elif formula == "Q2R2":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "α"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3])]}

    elif formula == "Q3R1":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1])]}

    elif formula == "Q3R2":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "α"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2])]}

    elif formula == "Q4R1":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b" , "h"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2])]}

    elif formula == "Q4R2":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "h", "α"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3])]}

    elif formula == "Q5R1":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b" , "a"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2])]}

    elif formula == "Q5R2":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "a", "α"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3])]}

    elif formula == "Q6R1":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3])]}

    elif formula == "Q6R2":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "α"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3]), 
                float('%.4g' % parameters[4])]}

    elif formula == "Q7R1":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "h"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3]), 
                float('%.4g' % parameters[4])]}

    elif formula == "Q7R2":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "h", "α"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3]), 
                float('%.4g' % parameters[4]), float('%.4g' % parameters[5])]}

    elif formula == "Q8R1":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "a"],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3]), 
                float('%.4g' % parameters[4])]}

    elif formula == "Q8R2":
        stats = {" ": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "a", "α", ],
                f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), float('%.4g' % aicc),
                float('%.4g' % parameters[0]), float('%.4g' % parameters[1]),
                float('%.4g' % parameters[2]), float('%.4g' % parameters[3]), 
                float('%.4g' % parameters[4]), float('%.4g' % parameters[5])]}

    else:
        raise ValueError("Not a valid formula; Only valid are Q[1-8]*R[1-2]*.")

    stats_df = pd.DataFrame(stats).set_index(" ")
    return stats_df