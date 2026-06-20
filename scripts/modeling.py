# scripts/modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

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

# =========== Анализ коэффициентов =================
def analyze_coefficients(model: LinearRegression, X_train: pd.DataFrame, 
                         save_path: str = None) -> pd.DataFrame:
    """Анализ коэффициентов модели

    Args:
        model (LinearRegression): обученная модель линейной регрессии
        X_train (pd.DataFrame): DataFrame с признаками (для получения названий колонок)
        save_path (str, optional): путь для сохранения графика. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame с коэффициентами, отсортированный по абсолютному значению
    """
    
    # ЗАДАНИЕ: Создайте DataFrame с коэффициентами модели
    coef_df = pd.DataFrame({
        'Признак': X_train.columns,
        'Коэффициент': model.coef_
    })
    
    # Добавляем базовую зарплату (intercept) отдельной строкой
    intercept_df = pd.DataFrame({
        'Признак': ['intercept (базовая зарплата)'],
        'Коэффициент': [model.intercept_]
    })
    
    # ЗАДАНИЕ: Отсортируйте коэффициенты по абсолютному значению
    coef_df['Абсолютное значение'] = coef_df['Коэффициент'].abs()
    coef_df = coef_df.sort_values('Абсолютное значение', ascending=False)
    
    # ЗАДАНИЕ: Выведите топ-10 самых важных признаков
    print(f'\nТоп-10 самых важных признаков (по влиянию на зарплату):')
    top_10 = coef_df.head(10)
    for idx, row in top_10.iterrows():
        sign = '+' if row['Коэффициент'] > 0 else ''
        print(f" {row['Признак']:<35} {sign}{row['Коэффициент']:>7.2f}")
    
    # ЗАДАНИЕ: Постройте bar plot для топ-10 коэффициентов
    plt.figure(figsize=(10, 7))
    
    # Берем топ-10 и разворачиваем для удобства чтения
    plot_data = coef_df.head(10).iloc[::-1]
    
    # Цвета: зелёный для положительных, красный для отрицательных
    colors = ['green' if x > 0 else 'red' for x in plot_data['Коэффициент']]
    
    plt.barh(plot_data['Признак'], plot_data['Коэффициент'], color=colors, alpha=0.7)
    plt.xlabel('Коэффициент (влияние на зарплату, тыс. руб.)', fontsize=11)
    plt.title('Топ-10 признаков по влиянию на зарплату', fontsize=13, pad=15)
    plt.axvline(x=0, color='black', linewidth=0.8) # Вертикальная линия на 0
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Добавляем значения на столбцы
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        plt.text(
            row['Коэффициент'] + (2 if row['Коэффициент'] > 0 else -2),
            i,
            f"{row['Коэффициент']:.1f}",
            va='center',
            ha='left' if row['Коэффициент'] > 0 else 'right',
            fontsize=9
        )
    
    plt.tight_layout()
    
    # Сохраняем график
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'\nГрафик сохранён: {save_path}')
    
    plt.show()
    
    # Удаляем вспомогательную колонку перед возвратом
    coef_df = coef_df.drop(columns=['Абсолютное значение'])
    
    return coef_df

# ============ Проверка мультиколлинеарности через VIF ===============
def check_multicollinearity(X_train: pd.DataFrame) -> pd.DataFrame:
    """Проверка мультиколлинеарности через VIF

    Args:
        X_train (pd.DataFrame): DataFrame с признаками (обучающая выборка)

    Returns:
        pd.DataFrame: DataFrame с VIF для каждого признака
    """
    
    # Приводим данные к числовому типу float64
    X_numeric = X_train.astype(float)
    
    # Добавляем константу (столбец единиц) — это требование statsmodels
    # для корректного расчёта VIF
    from statsmodels.tools.tools import add_constant
    X_with_const = add_constant(X_numeric)
    
    # Вычисляем VIF
    vif_data = pd.DataFrame()
    vif_data["Признак"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i + 1)
                        for i in range(len(X_train.columns))]
    
    # ЗАДАНИЕ: Отсортируйте по VIF и выведите признаки с VIF > 10
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    
    # Выводим все признаки с VIF
    print('\n VIF для всех признаков:')
    for _, row in vif_data.iterrows():
        status = '+' if row['VIF'] < 5 else ('!' if row['VIF'] < 10 else 'X')
        print(f"  {status} {row['Признак']:<35} VIF = {row['VIF']:.2f}")
        
    # Выводим признаки с VIF > 10 (критическая мультиколлинеарность)
    critical_vif = vif_data[vif_data['VIF'] > 10]
    print(f'\n Признаки с критической мультиколлинеарностью (VIF > 10):')
    if len(critical_vif) > 0:
        for _, row in critical_vif.iterrows():
            print(f"   {row['Признак']}: VIF = {row['VIF']:.2f}")
    else:
        print(' Критической мультиколлинеарности не обнаружено!')
        
    # Выводим признаки с VIF > 5 (высокая мультиколлинеарность)
    high_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]
    print(f'\n Признаки с высокой мультиколлинеарностью (5 < VIF ≤ 10):')
    if len(high_vif) > 0:
        for _, row in high_vif.iterrows():
            print(f"  {row['Признак']}: VIF = {row['VIF']:.2f}")
    else:
        print('  Признаков с высокой мультиколлинеарностью не обнаружено!')
        
    return vif_data

# ============== Анализ значимости коэффициентов ============
def analyze_statistical_significance(X_train: pd.DataFrame, y_train: pd.Series) -> sm.regression.linear_model.RegressionResults:
    """Анализ значимости коэффициентов

    Args:
        X_train (pd.DataFrame): DataFrame с признаками (обучающая выборка)
        y_train (pd.Series): Series с целевой переменной (обучающая выборка)

    Returns:
        sm.regression.linear_model.RegressionResults: Обученная модель statsmodels OLS
    """
    
    # Приводим данные к числовому типу float64
    X_train_float = X_train.astype(float).reset_index(drop=True)
    y_train_float = y_train.astype(float).reset_index(drop=True)
    
    # ЗАДАНИЕ: Добавьте константу к признакам
    X_train_sm = sm.add_constant(X_train_float)
    
    # ЗАДАНИЕ: Обучите модель OLS из statsmodels
    model_stats = sm.OLS(y_train_float, X_train_sm).fit()
    
    # ЗАДАНИЕ: Выведите summary модели
    print('\n' + model_stats.summary().as_text())
    
    return model_stats

# ========== Отбор признаков (feature selection) на основе p-значений ============
def select_features_by_pvalue(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series, 
    y_test: pd.Series,
    pvalue_threshold: float = 0.05
) -> tuple:
    """Отбор признаков (feature selection) на основе p-значений

    Args:
        X_train (pd.DataFrame): DataFrame с признаками (обучающая выборка)
        X_test (pd.DataFrame): DataFrame с признаками (тестовая выборка)
        y_train (pd.Series): Series с целевой переменной (обучающая выборка)
        y_test (pd.Series): Series с целевой переменной (тестовая выборка)
        pvalue_threshold (float, optional): порог p-value для отбора. Defaults to 0.05.

    Returns:
        tuple: (X_train_selected, X_test_selected, model_selected, y_test_pred_selected)
    """
    
    # Приводим данные к float64 и сбрасываем индексы
    X_train_clean = X_train.astype(float).reset_index(drop=True)
    y_train_clean = y_train.astype(float).reset_index(drop=True)
    X_test_clean = X_test.astype(float).reset_index(drop=True)
    y_test_clean = y_test.astype(float).reset_index(drop=True)
    
    # Обучаем OLS модель для получения p-values
    X_train_sm = sm.add_constant(X_train_clean)
    model_stats = sm.OLS(y_train_clean, X_train_sm).fit()
    
    # ЗАДАНИЕ: Создайте список значимых признаков (p-value < 0.05)
    # Исключаем константу 'const' из списка
    p_values = model_stats.pvalues.drop('const')
    significant_features = p_values[p_values < pvalue_threshold].index.tolist()
    
    print(f'\nИсходное количество признаков: {len(X_train.columns)}')
    print(f'Порог p-value: {pvalue_threshold}')
    print(f'Количество значимых признаков: {len(significant_features)}')
    print(f'Количество удалённых признаков: {len(X_train.columns) - len(significant_features)}')
    
    print(f'\nЗначимые признаки (p < {pvalue_threshold}):')
    for feature in significant_features:
        p_val = p_values[feature]
        coef = model_stats.params[feature]
        print(f' - {feature:<35} p-value = {p_val:.4f}, коэф. = {coef:+.2f}')
        
    # ЗАДАНИЕ: Обучите модель только на значимых признаках
    X_train_selected = X_train_clean[significant_features]
    X_test_selected = X_test_clean[significant_features]
    
    model_selected = LinearRegression()
    model_selected.fit(X_train_selected, y_train_clean)
    
    # Предсказания
    y_train_pred_selected = model_selected.predict(X_train_selected)
    y_test_pred_selected = model_selected.predict(X_test_selected)
    
    return X_train_selected, X_test_selected, model_selected, y_test_pred_selected
