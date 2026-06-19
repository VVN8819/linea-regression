# scripts/feature_engineering.py
import pandas as pd
import numpy as np

# One-Hot Encoding для категориальных признаков
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Создаем копию датасета для обработки
    df_encoded = df.copy()
    
    # Категориальные признаки для кодирования
    categorical_columns = [
        'образование', 
        'город', 
        'язык_программирования',
        'размер_компании', 
        'английский'
    ]
    
    # ЗАДАНИЕ: Примените One-Hot Encoding для категориальных признаков
    df_encoded = pd.get_dummies(
        df_encoded, columns=categorical_columns, drop_first=True
    )
    
    # ЗАДАНИЕ: Выведите названия всех столбцов после кодирования
    print(f'\nНазвания всех столбцов после кодирования:')
    for col in df_encoded.columns:
        print(f'  - {col}')
    
    # ЗАДАНИЕ: Выведите форму датасета (количество строк и столбцов)
    print(f"\nКоличество строк: {df_encoded.shape[0]}")
    print(f"Количество столбцов: {df_encoded.shape[1]}")
    print(f"Новых признаков появилось: {df_encoded.shape[1] - df.shape[1]}")
    
    return df_encoded

# Разделение данных на признаки (X) и целевую переменную (y).
def prepare_features_and_target(df: pd.DataFrame) -> tuple:
    # Отделяем целевую переменную
    y = df['зарплата']
    
    # Все остальные колонки - признаки
    X = df.drop(columns=['зарплата'])
    
    print(f'\nФорма X (признаки): {X.shape}')
    print(f'Форма y (целевая переменная): {y.shape}')
    print(f'Целевая переменная: зарплата')
    print(f'Количество признаков: {X.shape[1]}')
    
    return X, y

# Пайплайн предобработки признаков
def run_feature_engineering(df: pd.DataFrame) -> tuple:
    
    # 1. Кодирование категориальных признаков
    df_encoded = encode_categorical_features(df)
    
    # 2. Разделение на X и y
    X, y = prepare_features_and_target(df_encoded)
    
    return X, y