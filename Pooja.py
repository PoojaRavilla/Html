numpy exercises:

#1.Create a NumPy array containing numbers from 1 to 10.

import numpy as np

arr = np.arange(1, 11)

print(arr)



#2.Reshape the array from exercise 1 into a 2x5 matrix.

reshaped_arr = arr.reshape(2, 5)

print(reshaped_arr)


#3.Create a 3x3 identity matrix using NumPy.

identity_matrix = np.eye(3)

print(identity_matrix)



#Pandas Exercises:

#1. Create a DataFrame from a dictionary containing student names and their corresponding

scores.

import pandas as pd

data = {'Student': ['Alice', 'Bob', 'Charlie'], 'Score': [85, 90, 88], 'Age':[10,12,12]}

df = pd.DataFrame(data)

print(df)



#2.Select only the 'Student' and 'Age' columns from the DataFrame.
selected_columns = df[['Student', 'Age']]

print(selected_columns)

#3.Load a CSV file into a DataFrame.

#df = pd.read_csv('data.csv')

#print(df)

#Matplotlib Exercises:

#1. Plot a sine wave using Matplotlib.

import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(0, 2*np.pi, 100)

y = np.sin(x)

plt.plot(x, y)

plt.xlabel('x')

plt.ylabel('sin(x)')

plt.title('Sine Wave')

plt.show()
#2. Create a scatter plot of random data points.

x = np.random.rand(100)

y = np.random.rand(100)

plt.scatter(x, y)

plt.xlabel('X')plt.ylabel('Y')

plt.title('Scatter Plot')

plt.show()

#3. plot a histogram of normally distributed data.

data = np.random.randn(1000)

plt.hist(data, bins=30)

plt.xlabel('Value')

plt.ylabel('Frequency')

plt.title('Histogram of Normally Distributed Data')
plt.show()
