import pandas as pd

from scripts.data_loader import read_csv_data
from scripts.eda import run_eda


def main():
    # сырые данные
    df = read_csv_data()
    
    #print(df.head(10))
    
    run_eda(df)
    
if __name__ == "__main__":
    main()
    



