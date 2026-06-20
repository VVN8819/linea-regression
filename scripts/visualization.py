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
    plt.tight_layout()
    
    # Сохраняем график
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'График сохранён: {save_path}')
    
    plt.show()

def find_outliers_iqr(df: pd.DataFrame, col: str) -> int:
    
    print(f'\nАнализ выбросов в колонке: {col}')
    
    data = df[col]
    
    # ========================== Ищем выбросы (метод IQR) ========================
    Q1 = data.quantile(0.25)  # Нижний квартиль
    Q3 = data.quantile(0.75)  # Верхний квартиль
    IQR = Q3 - Q1  # межквартильный размах
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_pct = len(outliers) / len(data) * 100

    print(f'\nВыбросы в {col}:')
    print(f'Нижний квартиль: {Q1:.2f}')
    print(f'Верхний квартиль: {Q3:.2f}')
    print(f'Межквартильный размах: {IQR:.2f}')
    print(f'Верхние и нижние границы исключения выбросов: [{lower_bound:,.0f}; {upper_bound:,.0f}]')
    print(f'Выбросов: {len(outliers)} шт. ({outlier_pct:.1f}%)')

    if len(outliers) > 0 and len(outliers) <= 5:
        print(f'Значения: {outliers.tolist()}')
    elif len(outliers) > 5:
        print(f'Примеры: {outliers.head(3).tolist()} ...')
        
    # Дополнительная информация о выбросах
    if len(outliers) > 0:
        print(f'\nДетали выбросов:')
        outlier_details = df.loc[outliers.index]
        print(outlier_details[['опыт_лет', 'возраст', 'город', 'язык_программирования', 'зарплата']].to_string())

    return len(outliers)


def run_visualizations(df: pd.DataFrame, reports_dir: str = 'scripts/reports') -> None:
    
    # Формируем пути для сохранения
    salary_dist_path = f'{reports_dir}/salary_distribution.png'
    exp_vs_salary_path = f'{reports_dir}/experience_vs_salary.png'
    
    # Создаём папку reports, если её нет
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    # Анализ выбросов
    find_outliers_iqr(df, 'зарплата')
    
    plot_salary_distribution(df, save_path=salary_dist_path)
    plot_experience_vs_salary(df, save_path=exp_vs_salary_path)



