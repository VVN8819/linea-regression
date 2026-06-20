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
    
    # ЗАДАНИЕ: Сравните метрики с базовой моделью
    # Вычисляем метрики для новой модели
    r2_train_selected = r2_score(y_train_clean, y_train_pred_selected)
    r2_test_selected = r2_score(y_test_clean, y_test_pred_selected)
    rmse_train_selected = np.sqrt(mean_squared_error(y_train_clean, y_train_pred_selected))
    rmse_test_selected = np.sqrt(mean_squared_error(y_test_clean, y_test_pred_selected))
    mae_train_selected = mean_absolute_error(y_train_clean, y_train_pred_selected)
    mae_test_selected = mean_absolute_error(y_test_clean, y_test_pred_selected)
    
    # Получаем метрики базовой модели для сравнения
    model_baseline = LinearRegression()
    model_baseline.fit(X_train_clean, y_train_clean)
    y_train_pred_baseline = model_baseline.predict(X_train_clean)
    y_test_pred_baseline = model_baseline.predict(X_test_clean)
    
    r2_train_baseline = r2_score(y_train_clean, y_train_pred_baseline)
    r2_test_baseline = r2_score(y_test_clean, y_test_pred_baseline)
    rmse_train_baseline = np.sqrt(mean_squared_error(y_train_clean, y_train_pred_baseline))
    rmse_test_baseline = np.sqrt(mean_squared_error(y_test_clean, y_test_pred_baseline))
    mae_train_baseline = mean_absolute_error(y_train_clean, y_train_pred_baseline)
    mae_test_baseline = mean_absolute_error(y_test_clean, y_test_pred_baseline)
    
    # Сравниваем метрики
    print('\nБазовая модель vs С отбором признаков\n')
    print(f'{"Метрика":<25} {"Базовая":<15} {"С отбором":<15} {"Разница":<15}\n')
    print(f'{"R² (train)":<25} {r2_train_baseline:<15.4f} {r2_train_selected:<15.4f} {r2_train_selected - r2_train_baseline:>+15.4f}')
    print(f'{"R² (test)":<25} {r2_test_baseline:<15.4f} {r2_test_selected:<15.4f} {r2_test_selected - r2_test_baseline:>+15.4f}')
    print(f'{"RMSE (train), тыс. руб.":<25} {rmse_train_baseline:<15.2f} {rmse_train_selected:<15.2f} {rmse_train_selected - rmse_train_baseline:>+15.2f}')
    print(f'{"RMSE (test), тыс. руб.":<25} {rmse_test_baseline:<15.2f} {rmse_test_selected:<15.2f} {rmse_test_selected - rmse_test_baseline:>+15.2f}')
    print(f'{"MAE (train), тыс. руб.":<25} {mae_train_baseline:<15.2f} {mae_train_selected:<15.2f} {mae_train_selected - mae_train_baseline:>+15.2f}')
    print(f'{"MAE (test), тыс. руб.":<25} {mae_test_baseline:<15.2f} {mae_test_selected:<15.2f} {mae_test_selected - mae_test_baseline:>+15.2f}')
    
    # Интерпретация
    print('\nИнтерпретация:')
    r2_diff = r2_test_selected - r2_test_baseline
    if r2_diff > 0.01:
        print(f'Модель улучшилась! R² на тесте вырос на {r2_diff:.4f}')
    elif r2_diff < -0.01:
        print(f'Модель ухудшилась. R² на тесте упал на {abs(r2_diff):.4f}')
    else:
        print(f'Модель практически не изменилась (разница R² = {r2_diff:.4f})')
    
    mae_diff = mae_test_selected - mae_test_baseline
    print(f'Изменение MAE на тесте: {mae_diff:+.2f} тыс. руб.')
    
    print(f'\nУпрощение модели:')
    print(f'Было признаков: {len(X_train.columns)}')
    print(f'Стало признаков: {len(significant_features)}')
    print(f'Сокращение: {((len(X_train.columns) - len(significant_features)) / len(X_train.columns) * 100):.1f}%')
    
    return X_train_selected, X_test_selected, model_selected, y_test_pred_selected, r2_test_selected, mae_test_selected

# =============== Бизнес-выводы ===================
def business_insights(model_selected: LinearRegression, 
                      X_train_selected: pd.DataFrame,
                      r2_test: float = None,
                      mae_test: float = None) -> None:
    """Бизнес-выводы на основе коэффициентов финальной модели

    Args:
        model_selected (LinearRegression): обученная модель после отбора признаков
        X_train_selected (pd.DataFrame): DataFrame с признаками финальной модели
        r2_test (float, optional): R² на тестовой выборке. Defaults to None.
        mae_test (float, optional): MAE на тестовой выборке. Defaults to None.
    """
    
    # Создаем DataFrame с коэффициентами для удобного доступа
    coef_df = pd.DataFrame({
        'Признак': X_train_selected.columns,
        'Коэффициент': model_selected.coef_
    })
    
    # Базовая зарплата (intercept)
    base_salary = model_selected.intercept_
    print(f'\nБазовая зарплата (intercept): {base_salary:.2f} тыс. руб.')
    
    # ===== 1: На сколько увеличивается зарплата с каждым годом опыта? =====
    exp_coef = coef_df[coef_df['Признак'] == 'опыт_лет']['Коэффициент'].values[0]
    print(f'\nОтвет: +{exp_coef:.2f} тыс. руб. за каждый год опыта')
    print(f'\nПримеры:')
    print(f'   • Junior (0 лет опыта): {base_salary:.0f} тыс. руб.')
    print(f'   • Middle (5 лет опыта): {base_salary + 5 * exp_coef:.0f} тыс. руб.')
    print(f'   • Senior (10 лет опыта): {base_salary + 10 * exp_coef:.0f} тыс. руб.')
    print(f'   • Lead (14 лет опыта): {base_salary + 14 * exp_coef:.0f} тыс. руб.')
    
    # ===== 2: Какой город дает самую высокую премию к зарплате? =====
     
    # Базовый город — Казань (когда все город_* = 0)
    city_features = [col for col in coef_df['Признак'] if col.startswith('город_')]
    city_coefs = coef_df[coef_df['Признак'].isin(city_features)]
    
    print(f'\nБазовый город (для сравнения): Казань')
    print(f'\nНадбавки к зарплате по городам:')
    print(f'   • Казань (базовый): 0 тыс. руб.')
    
    best_city = None
    best_city_coef = 0
    for _, row in city_coefs.iterrows():
        city_name = row['Признак'].replace('город_', '')
        print(f'   • {city_name}: {row["Коэффициент"]:+.2f} тыс. руб.')
        if row['Коэффициент'] > best_city_coef:
            best_city_coef = row['Коэффициент']
            best_city = city_name
    
    print(f'\nОтвет: {best_city} дает самую высокую премию (+{best_city_coef:.2f} тыс. руб.)')
    
    # ===== 3: Какой язык программирования самый выгодный? =====
    
    # Базовый язык — C++ (когда все язык_* = 0)
    lang_features = [col for col in coef_df['Признак'] if col.startswith('язык_программирования_')]
    lang_coefs = coef_df[coef_df['Признак'].isin(lang_features)]
    
    print(f'\nБазовый язык (для сравнения): C++')
    print(f'\nРазница в зарплате по языкам:')
    print(f'   • C++ (базовый): 0 тыс. руб. (самый высокооплачиваемый)')
    
    for _, row in lang_coefs.iterrows():
        lang_name = row['Признак'].replace('язык_программирования_', '')
        print(f'   • {lang_name}: {row["Коэффициент"]:+.2f} тыс. руб.')
        
    print(f'\nВывод: JavaScript-разработчики получают на {abs(lang_coefs[lang_coefs["Признак"]=="язык_программирования_JavaScript"]["Коэффициент"].values[0]):.0f} тыс. руб.')

    # ===== 4: Сколько стоит знание английского на уровне C1-C2 vs A1-A2? =====
    
    # Базовый уровень — A1-A2 (когда все английский_* = 0)
    eng_features = [col for col in coef_df['Признак'] if col.startswith('английский_')]
    eng_coefs = coef_df[coef_df['Признак'].isin(eng_features)]
    
    c1_c2_coef = eng_coefs[eng_coefs['Признак'] == 'английский_C1-C2']['Коэффициент'].values[0]
    b1_b2_coef = eng_coefs[eng_coefs['Признак'] == 'английский_B1-B2']['Коэффициент'].values[0]
    
    print(f'\nНадбавки за знание английского:')
    print(f'   • A1-A2 (базовый): 0 тыс. руб.')
    print(f'   • B1-B2 (средний): +{b1_b2_coef:.2f} тыс. руб.')
    print(f'   • C1-C2 (свободное владение): +{c1_c2_coef:.2f} тыс. руб.')
    
    print(f'\nОтвет: C1-C2 vs A1-A2 = +{c1_c2_coef:.2f} тыс. руб.')
    print(f'\nВывод:')
    print(f'   • Переход с A1-A2 на B1-B2 даёт +{b1_b2_coef:.0f} тыс. руб.')
    print(f'   • Переход с B1-B2 на C1-C2 даёт ещё +{c1_c2_coef - b1_b2_coef:.0f} тыс. руб.')
    print(f'   • Общий переход с A1-A2 на C1-C2 даёт +{c1_c2_coef:.0f} тыс. руб.')
    print(f'   • Это {c1_c2_coef/exp_coef:.1f} лет опыта работы!')
    print(f'\nРекомендация кандидатам: учите английский до C1-C2 — это')
    print(f'   окупается так же, как {c1_c2_coef/exp_coef:.0f} лет дополнительного опыта.')
    
    # ===== ИТОГОВЫЕ БИЗНЕС-РЕКОМЕНДАЦИИ =====
    print('ИТОГОВЫЕ БИЗНЕС-РЕКОМЕНДАЦИИ')
    
    print('\nДля кандидатов (как максимизировать зарплату):')
    print('   1. Накапливайте опыт — каждый год = +' + f'{exp_coef:.0f}' + ' тыс. руб.')
    print('   2. Учите английский до C1-C2 — надбавка +' + f'{c1_c2_coef:.0f}' + ' тыс. руб.')
    print('   3. Стремитесь в крупные компании — надбавка до +' + f'{abs(coef_df[coef_df["Признак"]=="размер_компании_Малая"]["Коэффициент"].values[0]):.0f}' + ' тыс. руб.')
    print('   4. Рассмотрите переезд в Москву — надбавка +' + f'{best_city_coef:.0f}' + ' тыс. руб.')
    print('   5. Получите PhD или Магистратуру — Бакалавр стоит -31 тыс. руб.')
    
    print('\nДля работодателей (рыночные ставки):')
    print('   • Базовая ставка (0 опыта, Казань, C++, Крупная, A1-A2, PhD): ' + f'{base_salary:.0f}' + ' тыс. руб.')
    print('   • Москва добавляет +' + f'{best_city_coef:.0f}' + ' тыс. руб. к любому кандидату')
    print('   • Малая компания должна платить на ~49 тыс. руб. меньше крупной')
    print('   • C1-C2 английский стоит +' + f'{c1_c2_coef:.0f}' + ' тыс. руб. в месяц')
    
    print('\nДля HR-аналитики:')
    if r2_test is not None and mae_test is not None:
        print(f'   • Модель объясняет {r2_test*100:.1f}% различий в зарплатах на тестовой выборке')
        print(f'   • Средняя ошибка предсказания: {mae_test:.1f} тыс. руб.')
    else:
        print('   • Модель использует 10 статистически значимых факторов')
    print('   • 10 ключевых факторов определяют зарплату разработчика')
    
    
    