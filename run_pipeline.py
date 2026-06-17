import pandas as pd

from scripts.data_loader import read_csv_data

def main():
    # сырые данные
    df = read_csv_data()
    
    print(df.head(10))
    
if __name__ == "__main__":
    main()
    



