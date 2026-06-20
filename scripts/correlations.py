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

# ======== Строим тепловую карту (heatmap) корреляционной матрицы ===============
def plot_correlation_heatmap(df_encoded: pd.DataFrame, save_path: str = None) -> None:
    """Построение тепловой карты (heatmap) корреляционной матрицы
    для числовых столбцов.

    Args:
        df_encoded (pd.DataFrame): DataFrame с данными
        save_path (str, optional): путь для сохранения графика. Defaults to None.
    """
    
    # ЗАДАНИЕ: Постройте тепловую карту (heatmap) корреляционной матрицы
    # для числовых столбцов
    numeric_cols = ['опыт_лет', 'возраст', 'образование_код', 'зарплата']
    df_numeric = df_encoded[numeric_cols]
    
    # Вычисляем корреляционную матрицу
    corr_matrix = df_numeric.corr()
    print(f'\nКорреляционная матрица (числовые признаки):\n {corr_matrix.round(3)}')
    
    # Строим heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True, # Показываем значения на карте
        cmap='coolwarm',
        center=0,
        fmt='.3f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    plt.title('Корреляционная матрица числовых признаков', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Сохраняем график
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'\nГрафик сохранён: {save_path}')
    
    plt.show()

def run_correlations(df_encoded: pd.DataFrame, reports_dir: str = 'scripts/reports') -> None:
    """Запуск полного корреляционного анализа.

    Args:
        df_encoded (pd.DataFrame): DataFrame с закодированными признаками
        reports_dir (str, optional): папка для сохранения отчётов. Defaults to 'scripts/reports'.
    """
    # 1. Анализ корреляции с целевой переменной
    correlations = analyze_correlations_with_target(df_encoded)
    
    # 2. Построение heatmap
    heatmap_path = f'{reports_dir}/correlation_heatmap.png'
    plot_correlation_heatmap(df_encoded, save_path=heatmap_path)
    
    return correlations

