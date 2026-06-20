# scripts/modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ========== Разделите данные на признаки (X) и целевую переменную (y) =============
def prepare_and_split(df_encoded: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Подготовка признаков/целевой переменной и разделение на train/test.

    Args:
        df_encoded (pd.DataFrame): DataFrame с закодированными признаками и целевой переменной
        test_size (float, optional): доля тестовой выборки. Defaults to 0.2.
        random_state (int, optional): фиксация случайности для воспроизводимости. Defaults to 42.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # ЗАДАНИЕ: Разделите данные на признаки (X) и целевую переменную (y)
    # колонки - признаки
    X = df_encoded.drop('зарплата', axis=1)
    # Отделяем целевую переменную
    y = df_encoded['зарплата']
    
    # Удаляем также служебный столбец 'образование_код' если он еще остался
    X = X.drop('образование_код', axis=1, errors='ignore')
    
    print(f"\nИтоговое количество признаков для модели: {X.shape[1]}")
    print(f"Названия признаков: {X.columns.tolist()}")
    
    # ЗАДАНИЕ: Разделите данные на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test