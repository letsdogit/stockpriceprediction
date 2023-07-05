import pandas as pd

# Load the dataset
dataset_url = 'https://drive.google.com/uc?id=190t0KiKqSdbFl-o_6r3S9Tvwo2mHzrcB'
df = pd.read_csv(dataset_url)

# Perform Univariate Analysis
# Example: Histogram of a numerical column
df['Age'].plot.hist(bins=20)

# Perform Bi-Variate Analysis
# Example: Scatter plot of two numerical columns
df.plot.scatter(x='Age', y='Salary')

# Perform Multi-Variate Analysis
# Example: Pairwise scatter plot of multiple numerical columns
pd.plotting.scatter_matrix(df[['Age', 'Salary', 'Experience']])

# Perform descriptive statistics on the dataset
statistics = df.describe()
print(statistics)

# Handle missing values
# Example: Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Find outliers and replace them (assuming 'Age' column has outliers)
# Example: Replace outliers with the median of the column
median_age = df['Age'].median()
q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
df['Age'] = df['Age'].apply(lambda x: median_age if x > upper_bound or x < lower_bound else x)

# Check for categorical columns and perform encoding
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Split the data into dependent and independent variables
X = df_encoded.drop('Target_variable', axis=1)
y = df_encoded['Target_variable']

# Scale the independent variables
# Example: Using StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
# Example: Using train_test_split from scikit-learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Please note that you need to replace 'Target_variable' with the actual name of the target variable column in the dataset.