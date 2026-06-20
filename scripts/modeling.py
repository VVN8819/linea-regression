# scripts/modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

# ===== Обучение базовой модели ============
def train_baseline_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame
    ) -> tuple:
    """Обучение базовой модели линейной регрессии и оценка метрик

    Args:
        X_train (pd.DataFrame): обучающей признаки
        X_test (pd.DataFrame): тестовые признаки
        y_train (pd.DataFrame): обучающая целевая переменная
        y_test (pd.DataFrame): тестовая целевая переменная

    Returns:
        tuple: model, y_test_pred предсказание
    """
    
    # ЗАДАНИЕ: Создайте и обучите модель линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # ЗАДАНИЕ: Сделайте предсказания на обучающей и тестовой выборках
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # ЗАДАНИЕ: Вычислите метрики качества R², RMSE, MAE для train и test
    # R² (Коэффициент детерминации)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    # RMSE (Среднеквадратичная ошибка). 
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # MAE (Средняя абсолютная ошибка)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print("Метрики на обучающей выборке")
    print(f"R²: {r2_train:.4f}")
    print(f"RMSE: {rmse_train:.2f} тыс. руб.")
    print(f"MAE: {mae_train:.2f} тыс. руб.")
    
    print("\nМетрики на тестовой выборке")
    print(f"R²: {r2_test:.4f}")
    print(f"RMSE: {rmse_test:.2f} тыс. руб.")
    print(f"MAE: {mae_test:.2f} тыс. руб.")
    
    print(f"\nМодель объясняет {r2_test*100:.1f}% различий в зарплатах на новых данных.")
    print(f"В среднем модель ошибается при предсказании на {mae_test:.1f} тыс. руб.")
    
    return model, y_test_pred
