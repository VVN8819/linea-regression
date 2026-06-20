import pandas as pd

from scripts.data_loader import read_csv_data
from scripts.eda import run_eda
from scripts.visualization import run_visualizations
from scripts.feature_engineering import run_feature_engineering
from scripts.correlations import run_correlations

def main():
    # сырые данные
    df = read_csv_data()
    
    # Запуск исследовательского анализа
    run_eda(df)
    
    # Визуализация распределений
    run_visualizations(df)
    
    # Предобработка признаков (One-Hot Encoding)
    X, y = run_feature_engineering(df)
    
    # Корреляционный анализ
    df_encoded = X.copy()
    df_encoded['зарплата'] = y
    correlations = run_correlations(df_encoded)
    
if __name__ == "__main__":
    main()
    



