# scripts/visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_salary_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    
    plt.figure(figsize=(12, 5))
    
    # ЗАДАНИЕ: Постройте гистограмму распределения зарплат
    plt.subplot(1, 2, 1)
    salaries = df['зарплата']
    plt.hist(salaries, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    salaries_median = salaries.median()
    plt.axvline(salaries_median, color='red', linestyle='--',
                label=f'Медиана: {salaries_median:.1f}')
    plt.title('Распределение зарплат')
    plt.xlabel('Зарплата (тыс. руб.)')
    plt.ylabel('Количество разработчиков')
    plt.legend()
    
    # ЗАДАНИЕ: Постройте boxplot для зарплат, чтобы увидеть выбросы
    plt.subplot(1, 2, 2)
    plt.boxplot(salaries, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Boxplot зарплат (выбросы)')
    plt.xlabel('Зарплата (тыс. руб.)')
    plt.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Анализ зарплат', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Сохраняем график
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'График сохранён: {save_path}')
    
    plt.show()

def plot_experience_vs_salary(df: pd.DataFrame, save_path: str = None) -> None:
    
    # ЗАДАНИЕ: Постройте scatter plot: опыт работы vs зарплата
    plt.figure(figsize=(10, 6))
    plt.scatter(df['опыт_лет'], df['зарплата'], 
                alpha=0.7, color='teal', edgecolors='w', s=100)
    plt.title('Зависимость зарплаты от опыта работы')
    plt.xlabel('Опыт работы (лет)')
    plt.ylabel('Зарплата (тыс. руб.)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Сохраняем график
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'График сохранён: {save_path}')
    
    plt.show()

def run_visualizations(df: pd.DataFrame, reports_dir: str = 'scripts/reports') -> None:
    
    # Формируем пути для сохранения
    salary_dist_path = f'{reports_dir}/salary_distribution.png'
    exp_vs_salary_path = f'{reports_dir}/experience_vs_salary.png'
    
    # Создаём папку reports, если её нет
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    plot_salary_distribution(df, save_path=salary_dist_path)
    plot_experience_vs_salary(df, save_path=exp_vs_salary_path)



