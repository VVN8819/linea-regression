import pandas as pd

from scripts.data_loader import read_csv_data
from scripts.eda import run_eda
from scripts.visualization import run_visualizations

def main():
    # сырые данные
    df = read_csv_data()
    
    # Запуск исследовательского анализа
    run_eda(df)
    
    # Визуализация распределений
    run_visualizations(df)
    
if __name__ == "__main__":
    main()
    



