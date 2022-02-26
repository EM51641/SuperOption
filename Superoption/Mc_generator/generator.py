from ipaddress import v4_int_to_packed
from High_performance_fcts.mc_sims import BS_MC, Heston_MC
from Tools.Option_tools import OptionTools
from Garch.Garch_model import GARCH_MODEL_AR_1
import numpy as np 


def data_prep(data, horizon):
    returns = np.log(data/data.shift(1)).dropna()
    garch_parameters = GARCH_MODEL_AR_1(returns, horizon).garch
    return garch_parameters
