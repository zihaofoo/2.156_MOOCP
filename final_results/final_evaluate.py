import numpy as np
import scipy
import matplotlib.pyplot as plt
from linkage_utils import evaluate_submission, draw_mechanism, solve_mechanism, to_final_representation, evaluate_mechanism, is_pareto_efficient, get_population_csv, from_1D_representation, save_population_csv
from pymoo.indicators.hv import HV
import pandas as pd

evaluate_submission()