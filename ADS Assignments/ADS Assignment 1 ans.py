import pandas as pd
from datetime import datetime, timedelta

# 1. Printing name and age
name = "ChatGPT"
age = 3

print("Name:", name)
print("Age:", age)

# 2. Splitting the string
X = "Datascience is used to extract meaningful insights."
split_string = X.split()
print(split_string)

# 3. Function for multiplication of two numbers
def multiply(a, b):
    return a * b

result = multiply(5, 3)
print("Multiplication:", result)

# 4. Creating a dictionary of states and capitals
states = {
    "New York": "Albany",
    "California": "Sacramento",
    "Texas": "Austin",
    "Florida": "Tallahassee",
    "Illinois": "Springfield"
}

print("States:")
for state, capital in states.items():
    print(state, "->", capital)

# 5. Creating a list of 1000 numbers using the range function
number_list = list(range(1, 1001))
print(number_list)

# 6. Creating a 4x4 identity matrix
identity_matrix = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
print(identity_matrix)

# 7. Creating a 3x3 matrix with values ranging from 1 to 9
matrix = [[j + 1 + (i * 3) for j in range(3)] for i in range(3)]
print(matrix)

# 8. Creating two similar dimensional arrays and performing sum on them
array1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
array2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

result = [[array1[i][j] + array2[i][j] for j in range(len(array1[i]))] for i in range(len(array1))]
print(result)

# 9. Generating a series of dates from 1st Feb, 2023 to 1st March, 2023
start_date = datetime(2023, 2, 1)
end_date = datetime(2023, 3, 1)

dates = []
while start_date <= end_date:
    dates.append(start_date.strftime("%Y-%m-%d"))
    start_date += timedelta(days=1)

print(dates)

# 10. Converting a dictionary into a corresponding dataframe
dictionary = {'Brand': ['Maruti', 'Renault', 'Hyundai'], 'Sales': [250, 200, 240]}
df = pd.DataFrame(dictionary)
print(df)
