# Step 1: Download the dataset
# Download the dataset manually and save it as 'house_prices.csv' in your working directory.

# Step 2: Load the dataset
import pandas as pd

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Step 3: Perform Univariate Analysis
import matplotlib.pyplot as plt

# Example: Univariate analysis for 'Area' variable
plt.hist(df['Area'], bins=20)
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.title('Distribution of House Area')
plt.show()

# Repeat the above code for other variables to perform univariate analysis.

# Step 4: Perform Bi-Variate Analysis
import seaborn as sns

# Example: Bi-variate analysis for 'Area' and 'Price'
sns.scatterplot(data=df, x='Area', y='Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs. Price')
plt.show()

# Repeat the above code for other pairs of variables to perform bi-variate analysis.

# Step 5: Perform Multi-Variate Analysis
# Example: Multi-variate analysis using pair plot
sns.pairplot(df)
plt.show()

# Step 6: Perform Descriptive Statistics
# Example: Calculate descriptive statistics for the dataset
desc_stats = df.describe()
print(desc_stats)

# Step 7: Check for Missing Values and deal with them
# Example: Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# If there are missing values, decide on an appropriate strategy to handle them (e.g., imputation or removal).

# Step 8: Find and Handle Outliers
# Example: Box plot to identify outliers in 'Price' variable
sns.boxplot(data=df, x='Price')
plt.xlabel('Price')
plt.title('Box Plot of House Prices')
plt.show()

# Decide on an appropriate method to handle outliers (e.g., removal or transformation).

# Step 9: Check for Categorical Columns and perform encoding
# Example: Check for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print(categorical_cols)

# Perform encoding on categorical columns using techniques like one-hot encoding or label encoding.

# Step 10: Split the data into dependent and independent variables
X = df.drop('Price', axis=1)  # Independent variables
y = df['Price']  # Dependent variable

# Step 11: Scale the independent variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 12: Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 13: Build the Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# Step 14: Train the Model
model.fit(X_train, y_train)

# Step 15: Test the Model
y_pred = model.predict(X_test)

# Step 16: Measure the Performance using Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R-squared Score:', r2)
