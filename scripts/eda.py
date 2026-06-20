import pandas as pd
import numpy as np

# ======================== Исследовательский анализ EDA ============================
def basic_info(df: pd.DataFrame) -> None:
    
    # ЗАДАНИЕ: Выведите информацию о датасете (типы данных, пропуски)
    df.info(memory_usage='deep')
    
    # ЗАДАНИЕ: Выведите первые 10 строк датасета
    print(f'\nПервые 10 строк: \n{df.head(10)}')
    
    # 1. Сколько наблюдений (разработчиков) в датасете?
    print(f'\nДатасет содержит {df.shape[0]} наблюдений (разработчиков)')
    
    # 2. Сколько признаков (столбцов)?
    print(f'В датасете всего {df.shape[1]} признаков\n')
    
# 3. Есть ли пропущенные значения?
# выявим пропущенные значения с помощью .isnull() и посчитаем их количество через mean()
def missing_values_rep(df: pd.DataFrame) -> None:
    print(f'Выявим пропущенные значения:\n{df.isnull().mean()}\n')

# ======================= Описательная статистика ==================================
# Выводит описательную статистику только для числовых колонок.
# Числа округлены до 2 знаков
def descrip_stat(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(f'Описательная статистика для числовых колонок:\n {df[num_cols].describe().T.round(2)}')

def run_eda(df: pd.DataFrame) -> None:
    """Запуск полного цикла EDA.

    Args:
        df (pd.DataFrame): сырые данные
    """
    basic_info(df)
    missing_values_rep(df)
    descrip_stat(df)
    