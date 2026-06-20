import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'

# ========= Вычисляем корреляцию всех признаков с зарплатой ==============
def analyze_correlations_with_target(df_encoded: pd.DataFrame) -> pd.Series:
    """Вычислите корреляцию всех признаков с зарплатой

    Args:
        df_encoded (pd.DataFrame): с закодированными признаками

    Returns:
        pd.Series: Series с корреляциями, отсортированный по убыванию
    """
    # ЗАДАНИЕ: Вычислите корреляцию всех признаков с зарплатой
    correlations = df_encoded.corr()['зарплата']
    
    # ЗАДАНИЕ: Отсортируйте корреляции по убыванию и выведите топ-10
    correlations_sorted = correlations.sort_values(ascending=False)
    
    print(f'\nТоп-10 корреляций признаков с зарплатой: \n{correlations_sorted.head(10).round(3)}')
    
    return correlations_sorted
    
def run_correlations(df_encoded: pd.DataFrame, reports_dir: str = 'scripts/reports') -> None:
    """Запуск полного корреляционного анализа.

    Args:
        df_encoded (pd.DataFrame): DataFrame с закодированными признаками
        reports_dir (str, optional): папка для сохранения отчётов. Defaults to 'scripts/reports'.
    """
    # 1. Анализ корреляции с целевой переменной
    correlations = analyze_correlations_with_target(df_encoded)
    
    return correlations

