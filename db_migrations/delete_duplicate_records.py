from pymongo import MongoClient
import os
from typing import Dict, Optional, Any, Union
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

# 1. Setup
kwargs: Dict[str, str] = {}
if os.getenv('MONGODB_AWS_ACCESS_KEY'):
    kwargs["username"] = os.getenv('MONGODB_AWS_ACCESS_KEY') or ""
    kwargs["password"] = os.getenv('MONGODB_AWS_SECRET_KEY') or ""
    kwargs["authMechanism"] = "MONGODB-AWS"

mongo_uri = os.getenv('MONGODB_URI')
if not mongo_uri:
    raise ValueError("MONGODB_URI not found. Please set it in .env file.")

client: MongoClient = MongoClient(
    mongo_uri,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000,
    maxPoolSize=100,
    minPoolSize=10,
    **kwargs
)

# Use the same database name logic as mongodb_storage.py
db_name = os.getenv('MONGODB_NAME')
db_name = db_name if db_name else "causalflow"
db = client[db_name]
collection = db["runs"]

print(f"Connected to database: {db_name}, collection: runs")

# Verify connection and list available databases
try:
    # Test connection
    client.admin.command('ping')
    print("✓ MongoDB connection successful")
    
    # List databases
    db_list = client.list_database_names()
    print(f"Available databases: {db_list}")
    
    # Check if target database exists
    if db_name not in db_list:
        print(f"⚠ Warning: Database '{db_name}' not found in available databases!")
    
    # Count documents
    doc_count = collection.count_documents({})
    print(f"Documents in '{db_name}.runs': {doc_count}")
except Exception as e:
    print(f"✗ Connection error: {e}")
    exit(1)

# Query by run_id (the primary identifier used in mongodb_storage.py)
target_run_id = "run_SealQA_2025-12-18T02:41:05.742380"
# Try querying by run_id
print(f"Querying by run_id: {target_run_id}")
doc = collection.find_one({"run_id": target_run_id})

if not doc:
    print("Document not found!")
    exit()

print(f"Loaded Run: {doc.get('run_id')}")
print(f"Original Counts -> Passing: {len(doc.get('passing_traces', []))}, Failing: {len(doc.get('failing_traces', []))}")

def should_keep(trace):
    try:
        p_id_str = trace.get("problem_id", "")
        num_part = int(p_id_str.split("_")[-1])
        
        return num_part <= 239
    except (ValueError, IndexError, AttributeError):
        # Safety: Keep records that don't match the format to avoid accidental data loss
        return True

# 4. Filter the arrays
# We create new lists containing ONLY the items we want to keep
new_passing = [t for t in doc.get("passing_traces", []) if should_keep(t)]
new_failing = [t for t in doc.get("failing_traces", []) if should_keep(t)]

# 5. Calculate stats
removed_passing = len(doc.get("passing_traces", [])) - len(new_passing)
removed_failing = len(doc.get("failing_traces", [])) - len(new_failing)
total_removed = removed_passing + removed_failing

print(f"Plan to remove: {total_removed} traces (Passing: {removed_passing}, Failing: {removed_failing})")

# 6. Execute Update
if total_removed > 0:
    confirm = input("Type 'CONFIRM' to apply changes to DB: ")
    if confirm == "CONFIRM":
        # Use the same query that found the document
        query_filter: Dict[str, Union[str, ObjectId]] = {}
        query_filter = {"run_id": target_run_id}
        
        collection.update_one(
            query_filter,
            {
                "$set": {
                    "passing_traces": new_passing,
                    "failing_traces": new_failing,
                    "num_problems": len(new_passing) + len(new_failing)
                }
            }
        )
        print("Update successful.")
    else:
        print("Operation cancelled.")
else:
    print("No traces found with ID > 408.")