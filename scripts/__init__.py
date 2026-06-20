# scripts/__init__.py
"""Модули для анализа данных о зарплатах разработчиков."""

from .data_loader import read_csv_data
from .eda import run_eda
from .visualization import run_visualizations
from .feature_engineering import run_feature_engineering
from .correlations import run_correlations
from .modeling import prepare_and_split

__all__ = [
    'read_csv_data', 
    'run_eda', 
    'run_visualizations',
    'run_feature_engineering',
    'run_correlations',
    'prepare_and_split'
]