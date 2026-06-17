import pandas as pd

# Загрузка сырых данных из CSV
def read_csv_data(path='scripts/data/raw_data.csv'):
    
    df = pd.read_csv(path)
    return df