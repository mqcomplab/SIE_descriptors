# This scripts contains the base functions to run the scripts in the root directory

import pandas as pd
import numpy as np
from enum import Enum
from scipy.optimize import curve_fit
from adjustText import adjust_text
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression
from contextlib import suppress
import modules.formulas

class prop_col(Enum):
    """
    dG: Gibbs Free Energy, dH: Enthalpy, viscosity: Viscosity-B coefficients of electrolyte solutions per coordinating water molecule,
    diffusion: Diffusion coefficients of ions in water, gfet: Gibbs free energies of ion transfer from water to methanol,
    dlcst: ΔLCST of pNIPAM-coated silica particles in 1 M electrolyte solutions, sn2: SN2 reaction rate of iodomethane and ionic nucleophiles in methanol,
    activity: Activity of a human rhinovirus, lysosome: Temperature dependence of the cloud point of lysozyme.
    """
    dG = 6
    dH = 8
    viscosity = 9
    diffusion = 10
    gfet = 11
    dlcst = 12
    sn2 = 13
    activity = 14
    lysozyme = 15

def get_property_col(property):
    """ Get the column number of the property from the Enum class prop_col

    Args:
        property (str): List of all possible properties: dG, dH, viscosity, diffusion, gfet, dlcst, sn2, activity, lysozyme

    Returns:
        int: Column number of the property
    """
    return prop_col[property].value

def need_homo_1(formula):
    """ Inquire if formula requires E_HOMO-1 values?

    Args:
        formula (str): Q1R1 - Q8R2 models from `formula.py`

    Raises:
        ValueError: if it is not a valid formula

    Returns:
        bool: Is it a formula that requires E_HOMO-1 values?
    """
    homo_1_formulas = ["Q3R1","Q3R2","Q6R1","Q6R2"]
    not_homo_1_formulas = ["Q1R1", "Q1R2", "Q2R1", "Q2R2", "Q4R1", "Q4R2", "Q5R1", "Q5R2", "Q7R1", "Q7R2", "Q8R1", "Q8R2"]
    if any(i in formula for i in homo_1_formulas):
        return True
    elif any(i in formula for i in not_homo_1_formulas):
        return False
    else:
        raise ValueError("Not a valid formula; Only valid are Q[1-8]*R[1-2]*.")

def read_data(file_name, property, formula, by_charge):
    """ Load the dataset.
 
    Args:
        file_name (str): Files must be in CSV UTF-8 format to recognize chemical formulas.
        property (int): _description_
        formula (str): _description_
        by_charge (bool): Does data need to be separated by charge?

    Returns:
        df: pandas DataFrame (by_charge = True)
        df_list: list (by_charge = False)
    
    Notes: 
        - Files must be in CSV UTF-8 format to recognize chemical formulas.
        - All rows containing NaN values will be excluded from the new dataframe.
        - `by_charge = True` for generating a list of dataframes and plotting by charges. 
        - `by_charge = False` for generating one dataframe, stats tables, and plotting ALL charges.  
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
    """ Fits the parameters of a given formula function to a dataset.

    Parameters:
        formula (str): The name of the formula being used.
        formula_func (function): The formula function being used.
        property (str): The name of the property being fit.
        df (pandas DataFrame): The dataset to which the formula is being fit.

    Returns:
        prop (pandas Series): The property being fit.
        fit_data (numpy array): The fit data generated by the formula.
        parameters (numpy array): The optimized parameters of the formula function.
        SE (numpy array): The standard errors of the optimized parameters.

    Raises:
        TypeError: If the formula function or dataset is not of the correct type.
        ValueError: If the formula is not recognized or if the formula function does not match the formula name.

    Notes:
        - This function uses the SciPy `curve_fit` function to optimize the parameters of the given formula function.
        - If the formula requires E_HOMO-1 values, the function will include them in the optimization.
    """
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
    """ Plots a scatter plot of the fitted data against the actual data for a given property,
    along with a line of perfect agreement.

    Parameters:
        prop (pandas Series): The actual property values.
        fit_data (numpy array): The fitted property values.
        ax (matplotlib Axes object): The subplot on which to plot the data.
        df (pandas DataFrame): The dataset used to fit the property values.

    Returns:
        ax (matplotlib Axes object): The subplot with the plotted data.

    Notes:
        - This function uses the `scatter` method of the given Axes object to create the scatter plot,
          and the `plot` method to add the line of perfect agreement.
        - The function also adds text labels to the scatter plot representing the indices of the
          data points in the dataset used to fit the property values.
        - This function requires the `adjust_text` function from the `adjustText` package to be imported.
    """
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
    """ Plots a scatter plot of the fitted data against the actual data for a given property
    for either cations or anions, along with a line of best fit.

    Parameters:
        prop (pandas Series): The actual property values.
        fit_data (numpy array): The fitted property values.
        ax (matplotlib Axes object): The subplot on which to plot the data.
        ion_type (str): The type of ion to plot ("cations" or "anions").
        df (pandas DataFrame): The dataset used to fit the property values.

    Returns:
        ax (matplotlib Axes object): The subplot with the plotted data.

    Raises:
        ValueError: If `ion_type` is not "cations" or "anions".

    Notes:
        - This function uses the `scatter` method of the given Axes object to create the scatter plot,
          and the `plot` method to add the line of best fit.
        - The function also adds text labels to the scatter plot representing the indices of the
          data points in the dataset used to fit the property values.
        - The `ion_type` parameter determines whether the plot represents cations or anions.
        - This function requires the `adjust_text` function from the `adjustText` package to be imported.
    """
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
    """ Customize axes labels and title for different physical properties.

    Parameters:
        formula (str): Chemical formula of the compound.
        property (str) :Name of the physical property. View `class prop_cols(Enum)` for supported options.
        ax (matplotlib.axes.Axes): Axes object to be customized.

    Returns:
        ax (matplotlib.axes.Axes): Axes object with customized labels and title.

    Raises:
        ValueError: If `property` is not one of the supported options. 

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax = custom_axis("Q1R1", "dG", ax)
        >>> plt.show()
    """ 
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

    return ax

def get_errors(prop, fit_data, parameters):
    """ Calculates the mean absolute error (MAE), root-mean square error (RMSE), Akaike Information Criterion with correction for small samples (AICC),
    and RMSE of leave-one-out cross-validation (LOOCV) for a given property and its corresponding fit data.

    Parameters:
        prop (pandas Series): A 1-dimensional numpy array containing the actual property values.
        fit_data (np.ndarray): A 1-dimensional numpy array containing the fit data.
        parameters (np.ndarray): A 1-dimensional numpy array containing the parameters used for fitting from `curve_fit`.

    Returns:
        Tuple [float, float, float, float]: A tuple containing the MAE, RMSE, LOOCV RMSE and AICC, respectively.
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
    """ Calculates the minimum Akaike information criterion corrected for small samples (AICc) for a given property and charge.
    Parameters:

        property (str): The name of the property for which the AICc is to be calculated.
        file_name (str): The name of the file containing the data to be read.
        charge (int): The charge of ions.

    Returns:
        float: The minimum value of AICc obtained after fitting multiple linear regression models using different formulae.
    """
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
    """ Replace AICc values in stats_result DataFrame with the difference from the minimum AICc value.

    Parameters:
        aicc_list (list): AICc values for each regression model.
        stats_result (pandas.DataFrame): Results summary DataFrame with model statistics.

    Returns:
        stats_result (pandas.DataFrame): Updated DataFrame replaced with the AICc differences or dAICc.
    """
    if len(aicc_list) == 0:
        aicc_min = np.nan
    else:
        aicc_min = np.min(aicc_list)
    daicc_list = list(map(lambda v: v - aicc_min, aicc_list))
    stats_result.iloc[3]= ['%.2f' % elem for elem in daicc_list]
    return stats_result

def print_stats(formula, parameters, SE, mean_abs_err, rmse, loocv, aicc):
    """ Prints statistical results for a given formula, including mean absolute error (MAE), root-mean-square error (RMSE),
    leave-one-out cross-validation (LOOCV), and difference in Akaike information criterion corrected(dAICc).

    Parameters:
        formula (str): The formula used to fit the data.
        parameters (array-like): Array of the fitted parameters from `curve_fit`.
        SE (array-like): Array of the standard errors of the fitted parameters.
        mean_abs_err (float): The mean absolute error of the fitted data.
        rmse (float): The root-mean-square error of the fitted data.
        loocv (float): The leave-one-out cross-validation error of the fitted data.
        aicc (float): The Akaike information criterion corrected for finite sample sizes.

    Returns:
        stats_df (pandas.DataFrame): A DataFrame of the calculated statistics for the given formula.
    """   
    stats_dict = {
        "Q1R1": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b"],
        "Q1R2": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "α"],
        "Q2R1": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ"],
        "Q2R2": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "α"],
        "Q3R1": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b"],
        "Q3R2": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "α"],
        "Q4R1": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "h"],
        "Q4R2": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "h", "α"],
        "Q5R1": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b" , "a"],
        "Q5R2": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "a", "α"],
        "Q6R1": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ"],
        "Q6R2": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "α"],
        "Q7R1": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "h"],
        "Q7R2": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "h", "α"],
        "Q8R1": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "a"],
        "Q8R2": ["MAE", "RMSE", "LOOCV", "dAICc", "m", "b", "γ", "ξ", "a", "α"]
    }   
    stats = {" ": stats_dict[formula],
            f"{formula}": [float('%.4g' % mean_abs_err), float('%.4g' % rmse), float('%.4g' % loocv), 
            float('%.4g' % aicc)] + [float('%.4g' % p) for p in parameters]}

    stats_df = pd.DataFrame(stats).set_index(" ")
    return stats_df