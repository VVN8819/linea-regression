import pandas as pd

from scripts.data_loader import read_csv_data
from scripts.eda import run_eda
from scripts.visualization import run_visualizations
from scripts.feature_engineering import run_feature_engineering
from scripts.correlations import run_correlations
from scripts.modeling import (
    prepare_and_split,
    train_baseline_model,
    analyze_coefficients,
    check_multicollinearity,
    analyze_statistical_significance,
    select_features_by_pvalue
)

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
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = prepare_and_split(df_encoded)
    
    # Обучение базовой модели
    model, y_test_pred = train_baseline_model(X_train, X_test, y_train, y_test)
    
    # Анализ коэффициентов
    coef_df = analyze_coefficients(model, X_train, save_path='scripts/reports/coefficients_plot.png')
    
    # Проверка мультиколлинеарности - Вычисление VIF
    vif_df = check_multicollinearity(X_train)
    
    # Анализ значимости коэффициентов
    model_stats = analyze_statistical_significance(X_train, y_train)
    
    # Отбор признаков на основе p-значений
    X_train_selected, X_test_selected, model_selected, y_test_pred_selected = \
        select_features_by_pvalue(X_train, X_test, y_train, y_test, pvalue_threshold=0.05)
    
if __name__ == "__main__":
    main()
    



