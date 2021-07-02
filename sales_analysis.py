# Import necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

"""

# load All Sales Files
file_loc = "drive/MyDrive/Sales_Data/"
files = [x for x in os.listdir(file_loc)]

# Merging all sales data into a single file
all_data = pd.DataFrame()
for file in files:
  df = pd.read_csv(file_loc + file)
  all_data = pd.concat([all_data, df])
all_data.to_csv("sales_data.csv", index=False)

# Read in updated dataframe
sales_data = pd.read_csv("sales_data.csv")
sales_data.head()

"""

"""
Clean the data - remove all incompatible datas
Removed Datas:
 - NaN
 - Duplicated Columns
 - Convert neccessary Datatypes
"""

nan_df = sales_data[sales_data.isna().any(axis=1)]
sales_data = sales_data.dropna(how="all") # NaN
sales_data = sales_data[sales_data["Order Date"].str[0:2] != "Or"] # Duplicated Columns
sales_data["Quantity Ordered"] = pd.to_numeric(sales_data["Quantity Ordered"]) # To Int
sales_data["Price Each"] = pd.to_numeric(sales_data["Price Each"]) # To Float

# Adding Month Columns
sales_data['Month'] = sales_data["Order Date"].str[0:2]
sales_data["Month"] = sales_data["Month"].astype('int32')

## Add Sales Column
sales_data['Sales'] = sales_data["Quantity Ordered"] * sales_data['Price Each']

"""### Best Month for Sales and how much was earned in the following month"""

# Grouping datas
results = sales_data.groupby("Month").sum()

# Plotting the sales overview of each months
months = range(1,13)
plt.figure(figsize=(18,10))
plt.title("Sales Overview (Months)")
plt.xticks(months)
plt.xlabel("Months")
plt.ylabel("Sales ($) in millions")
plt.bar(months, results["Sales"])
plt.show()

"""### City with hightest number of sales"""

# New Column with City
def get_city(address):
  return address.split(",")[1]

def get_state(address):
  return address.split(",")[2].split(" ")[1]

sales_data["City"] = sales_data['Purchase Address'].apply(lambda x: get_city(x) + ", " + get_state(x))

#Grouping data with city
city_results = sales_data.groupby("City").sum()
cities = [city for city,df in sales_data.groupby("City")]

# Plot the sales of city
plt.figure(figsize=(18,10))
plt.title("Sales Overview (City)")
plt.bar(cities, city_results["Sales"])
plt.xticks(cities, rotation="vertical")
plt.xlabel("City")
plt.ylabel("Sales ($) in millions")
plt.show()

"""### What time should we display advertisemens to maximize the likelihood of customer’s buying product?"""

# Replacing order date into proper pandas datetime
sales_data['Order Date'] = pd.to_datetime(sales_data["Order Date"])

# Add custom time colums
sales_data["Hour"] = sales_data['Order Date'].dt.hour

# Group by hours
hours = [hour for hour, df in sales_data.groupby("Hour")]

plt.figure(figsize=(18,10))
plt.title("Peak Hour of Sales")
plt.xticks(hours)
plt.xlabel("Hours", size=18)
plt.ylabel("Count", size=18)
plt.plot(hours, sales_data.groupby("Hour").count())
plt.show()

# Recommended time for advertisement can be around 11 am or 7pm according to the graph

"""### Products that are often sold together"""

# New dataframe to keep track of duplicate products and combining them together
df = sales_data[sales_data["Order ID"].duplicated(keep=False)]

df["Grouped"] = df.groupby("Order ID")['Product'].transform(lambda x: ",".join(x))
df = df[['Order ID', "Grouped"]].drop_duplicates()

from itertools import combinations
from collections import Counter

count = Counter()

# Getting most items sold  together
for row in df["Grouped"]:
  row_list = row.split(",")
  count.update(Counter(combinations(row_list, 2)))
count.most_common(10)

"""## Products that sold the most?"""

# Group data by products and quantity ordered
product_group = sales_data.groupby("Product")
quantity_ordered = product_group.sum()["Quantity Ordered"]
products = [product for product,df in product_group]

# Plotting the graph
plt.figure(figsize=(18,10))
plt.title("Most Sold Products")
plt.bar(products, quantity_ordered)
plt.xticks(products, rotation="vertical")
plt.xlabel("Products")
plt.ylabel("Total Sales")
plt.show()

# Most Ordered/Sold product was AAA Batteries

prices = sales_data.groupby("Product").mean()["Price Each"]

fig, ax1 = plt.subplots()
fig.set_figwidth(18)
fig.set_figheight(10)
fig.suptitle("Most Sold Products with Prices")
ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='g')
ax2.plot(products, prices, "b")

ax1.set_xlabel("Products")
ax1.set_ylabel("Quantity Ordered", color="g")
ax2.set_ylabel("Price", color="b")
ax1.set_xticklabels(products, rotation="vertical")

plt.show()