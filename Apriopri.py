import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Initialize an empty list to store the dataset
dataset = []

# Read the CSV file and convert it into the desired format
with open('customer_transactions.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Append each row (transaction) to the dataset as a list
        dataset.append(row)

# Convert the transactions into a transaction encoding format
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm to find frequent item sets
frequent_itemsets = apriori(df, min_support=0.003, use_colnames=True)

# Sort the frequent item sets by support values
frequent_itemsets_sorted = frequent_itemsets.sort_values(by='support', ascending=False)

# Display the first 20 frequent item sets
print(frequent_itemsets_sorted.head(20))