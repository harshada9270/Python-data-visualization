import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from scipy.stats import pearsonr
from pandas import read_csv
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/Harshada/Downloads/swiggy.csv") #enter your file path

#last 10 entries
last_x_rows = df.tail(10)
print("last 10 entries:",last_x_rows)

#first 10 entries
first_x_rows=df.head(10)
print("first 10 entries:",first_x_rows)

# Assuming the dataset is a CSV file named "swiggy.csv" in your Downloads folder

#plot scatter plot
# Create a scatter plot (assuming columns named 'Price' and 'Avg ratings')
colors = ['blue', 'green'] #color of the dots
plt.scatter(df['Price'], df['Avg ratings'])

# Add labels and title
plt.xlabel('Price')
plt.ylabel('Avg Rating')
plt.title('scatter plot')
plt.show()

#histogram 
plt.hist(df['Delivery time'])
plt.title('Histogram')

#box plot
df.boxplot(by='Price', column="Avg ratings", grid=False)
plt.title('box plot')
plt.title('box plot')
# Show the plot
plt.show()

#person correlation
list1=df["Price"]
list2=df["Avg ratings"]

corr,_=pearsonr(list1,list2)
print("person correlation is:",corr)

#independent variable=price
#dependant variable=avg ratings


#linear relationship
X = df[['Price']].values.reshape(-1,1) # Independent variable (Price)
y = df['Avg ratings'].values # Dependent variable (Avg ratings)

# Create a linear regression model
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Predict ratings based on the price
predicted_ratings = model.predict(X)

# Plot the original data
plt.scatter(df['Price'], df['Avg ratings'], color='blue', label='Actual Ratings')

# Plot the regression line (Predicted ratings)
plt.plot(df['Price'], predicted_ratings, color='red', label='Predicted Ratings')

plt.xlabel('Price')
plt.ylabel('Avg Ratings')
plt.title('Price vs Avg Ratings with Regression Line')
plt.legend()
plt.show()

# Print the model's slope and intercept
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)