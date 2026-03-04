# Импорт необходимых библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Загрузить данные с Kaggle
print("=" * 50)
print("ШАГ 1: Загрузка данных")
print("=" * 50)

df = pd.read_csv('train.csv')
print(f"Размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")

# 2. Вывести данные из датасета на экран
print("\n" + "=" * 50)
print("ШАГ 2: Вывод данных на экран")
print("=" * 50)
print(df.head(10))

# 3. Получить количество пропущенных значений для каждого столбца
print("\n" + "=" * 50)
print("ШАГ 3: Количество пропущенных значений")
print("=" * 50)
missing_values = df.isnull().sum()
print(missing_values)

# 4. Заполнить пропущенные значения
print("\n" + "=" * 50)
print("ШАГ 4: Заполнение пропущенных значений")
print("=" * 50)

# Создаем копию для обработки
df_filled = df.copy()

# Заполняем числовые пропуски медианой
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in numeric_cols:
    df_filled[col].fillna(df_filled[col].median(), inplace=True)

# Заполняем категориальные пропуски модой
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in categorical_cols:
    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

# Проверяем, что пропуски заполнены
print(f"Пропусков после заполнения: {df_filled.isnull().sum().sum()}")

# 5. Провести нормализацию данных
print("\n" + "=" * 50)
print("ШАГ 5: Нормализация данных")
print("=" * 50)

# Создаем копию для нормализации
df_normalized = df_filled.copy()

# MinMaxScaler
scaler = MinMaxScaler()
df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
print("Применен MinMaxScaler (значения в диапазоне 0-1)")

# 6. Преобразование категориальных данных (One-Hot Encoding)
print("\n" + "=" * 50)
print("ШАГ 6: One-Hot Encoding категориальных данных")
print("=" * 50)

# Применяем OHE с drop_first=True для избежания переобучения
df_final = pd.get_dummies(df_normalized, columns=categorical_cols, drop_first=True)

print(f"Размер датасета до OHE: {df_normalized.shape}")
print(f"Размер датасета после OHE: {df_final.shape}")

# Сохраняем обработанные данные
df_final.to_csv("processed_spaceship.csv", index=False)
print("\nФайл processed_spaceship.csv сохранен")