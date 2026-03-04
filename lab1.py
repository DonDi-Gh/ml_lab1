
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("ШАГ 1: Загрузка данных")
print("=" * 50)

df = pd.read_csv('train.csv')
print(f"Размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")


print("\n" + "=" * 50)
print("ШАГ 2: Вывод данных на экран")
print("=" * 50)
print("Первые 10 строк исходных данных:")
print(df.head(10))

# Получить количество пропущенных 
print("\n" + "=" * 50)
print("ШАГ 3: Количество пропущенных значений")
print("=" * 50)
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])  # Показываем только столбцы с пропусками

# Заполнить пропущенные 
print("\n" + "=" * 50)
print("ШАГ 4: Заполнение пропущенных значений")
print("=" * 50)


df_filled = df.copy()

# Заполняем числовые пропуски медианой
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in numeric_cols:
    median_val = df_filled[col].median()
    df_filled[col].fillna(median_val, inplace=True)
    #print(f"Колонка {col} заполнена медианой: {median_val:.2f}")

# Заполняем категориальные пропуски модой
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in categorical_cols:
    mode_val = df_filled[col].mode()[0]
    df_filled[col].fillna(mode_val, inplace=True)
    #print(f"Колонка {col} заполнена модой: {mode_val}")


print(f"\nПропусков после заполнения: {df_filled.isnull().sum().sum()}")

# нормализация данных
print("\n" + "=" * 50)
print("ШАГ 5: Нормализация данных")
print("=" * 50)


df_normalized = df_filled.copy()


print("\nИсходные числовые значения (до нормализации):")
print(df_normalized[numeric_cols].head(10).round(2))


scaler = MinMaxScaler()
df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])



print("\nЧисловые значения после нормализации (первые 10 строк):")
print(df_normalized[numeric_cols].head(10).round(3))


print("\nСтатистика ДО нормализации:")
print(df_filled[numeric_cols].describe().round(2))

print("\nСтатистика ПОСЛЕ нормализации:")
print(df_normalized[numeric_cols].describe().round(3))

#  Преобразование категориальных данных (One-Hot Encoding)
print("\n" + "=" * 50)
print("ШАГ 6: One-Hot Encoding категориальных данных")
print("=" * 50)

# Показываем уникальные значения в категориальных колонках до OHE
print("\nУникальные значения в категориальных колонках ДО OHE:")
for col in categorical_cols:
    unique_vals = df_normalized[col].unique()
    print(f"{col}: {unique_vals}")

#  для избежания переобучения
df_final = pd.get_dummies(df_normalized, columns=categorical_cols, drop_first=True)



print(f"\nНазвания колонок после OHE (первые 20):")
print(df_final.columns.tolist()[:20])


print("\nПервые 10 строк финального обработанного датасета:")
print(df_final.head(10))

# Сохраняем обработанные данные
df_final.to_csv("processed_spaceship.csv", index=False)
print("\nФайл processed_spaceship.csv сохранен")


print("\n" + "=" * 50)
print("ИТОГОВАЯ ИНФОРМАЦИЯ")
print("=" * 50)
print(f"Исходный размер датасета: {df.shape}")
print(f"Конечный размер датасета: {df_final.shape}")
print(f"Количество пропусков в финальном датасете: {df_final.isnull().sum().sum()}")
print(f"Типы данных в финальном датасете:\n{df_final.dtypes.value_counts()}")