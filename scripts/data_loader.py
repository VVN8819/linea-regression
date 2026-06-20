import pandas as pd

# Загрузка сырых данных из CSV
def read_csv_data(path='scripts/data/raw_data.csv'):
    """Загрузка сырых данных из CSV.

    Args:
        path (str, optional): Источник сырых данных 'scripts/data/raw_data.csv'.

    Returns:
        df: сырые данные
    """
    df = pd.read_csv(path)
    return df