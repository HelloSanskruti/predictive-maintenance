import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv("sensor.csv") #loading data

df = df.drop_duplicates() #dropping duplicate rows 

target_col = 'Fault Detected'
df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

df.drop(columns=[ #dropping unnecessary columns
    'Sensor_ID',
    'Equipment_ID',
    'Failure Type',
    'Last Maintenance Date',
    'Maintenance Type',
    'Operational Status',
    'External Factors',
    'Equipment Relationship'
], errors='ignore', inplace=True)

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in numeric_features: #removing outliers
    std = X[col].std()
    mean = X[col].mean()
    X = X[(X[col] - mean).abs() <= 3 * std]
    y = y.loc[X.index]  # keep y in sync

numeric_transformer = Pipeline(steps=[ #building a pipeline for numeric features imputation and scaling
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='drop'  
)

X_processed = preprocessor.fit_transform(X) #applying transformations

X_processed_df = pd.DataFrame(X_processed, columns=numeric_features)
X_processed_df['Timestamp'] = df.loc[X.index, 'Timestamp'].values  # adding Timestamp back
X_processed_df[target_col] = y.values  # adding target back

df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce') # converting Timestamp to datetime

X_processed_df.to_csv("processed_sensor_data_datacamp.csv", index=False)
print("Preprocessing complete. File saved as 'processed_sensor_data_datacamp.csv'")
