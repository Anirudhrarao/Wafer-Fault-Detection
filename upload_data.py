# Importing libraries
from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# uniform resource indentifier
uri = "mongodb+srv://raorudhra16:3011@cluster0.wgsorsj.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# Create database and collection name
DATABASE_NAME = 'Project2'
COLLECTION_NAME = 'WaferFault'

# Read dataframe
df = pd.read_csv('notebooks\data\wafer_23012020_041211.csv')
df = df.drop("Unnamed: 0",axis=1)

# Convert data into json
json_records = list(json.loads(df.T.to_json()).values())

# Now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)