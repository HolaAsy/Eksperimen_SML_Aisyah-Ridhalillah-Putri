import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# Fungsi preprocessing lengkap
def preprocessing_pipeline(csv_path):

    # Load dataset
    df = pd.read_csv(csv_path)

    # Hapus kolom yang tidak digunakan
    df = df.drop(columns=['Customer ID', 'Promo Code Used'])

    # Hapus data duplikat
    df = df.drop_duplicates().reset_index(drop=True)

    # Scaling data numerikal
    numerical_columns = [
        'Age',
        'Purchase Amount (USD)',
        'Review Rating',
        'Previous Purchases'
    ]

    def scaling(features, df):

        for feature in features:
            scaler = MinMaxScaler()

            scaler.fit(df[[feature]])

            df[feature] = scaler.transform(df[[feature]])

        return df

    df = scaling(numerical_columns, df)

    # Encoding fitur kategorikal
    categorical_columns = [
        'Gender',
        'Item Purchased',
        'Category',
        'Location',
        'Size',
        'Color',
        'Season',
        'Subscription Status',
        'Shipping Type',
        'Discount Applied',
        'Payment Method',
        'Frequency of Purchases'
    ]

    def encoding(features, df):

        for feature in features:
            encoder = LabelEncoder()

            encoder.fit(df[feature])

            df[feature] = encoder.transform(df[feature])

        return df

    df = encoding(categorical_columns, df)

    # Return hasil preprocessing
    return df


current_dir = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(
    current_dir,
    "..",
    "shopping_trends_raw.csv"
)

output_path = os.path.join(
    current_dir,
    "shopping_trends_preprocessing.csv"
)

# Jalankan preprocessing
df_processed = preprocessing_pipeline(input_path)

# Simpan hasil preprocessing
df_processed.to_csv(
    output_path,
    index=False
)

print("Preprocessing selesai.")
print(f"Dataset input  : {input_path}")
print(f"Dataset output : {output_path}")
print("\n5 Data Teratas:")
print(df_processed.head())