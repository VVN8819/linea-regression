# scripts/visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_salary_distribution(df: pd.DataFrame) -> None:
    
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
    plt.show()

def plot_experience_vs_salary(df: pd.DataFrame) -> None:
    
    # ЗАДАНИЕ: Постройте scatter plot: опыт работы vs зарплата
    plt.figure(figsize=(10, 6))
    plt.scatter(df['опыт_лет'], df['зарплата'], 
                alpha=0.7, color='teal', edgecolors='w', s=100)
    plt.title('Зависимость зарплаты от опыта работы')
    plt.xlabel('Опыт работы (лет)')
    plt.ylabel('Зарплата (тыс. руб.)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def run_visualizations(df: pd.DataFrame) -> None:
    
    plot_salary_distribution(df)
    plot_experience_vs_salary(df)



