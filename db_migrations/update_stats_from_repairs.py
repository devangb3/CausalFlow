from pymongo import MongoClient
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# 1. Setup MongoDB connection
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

# Verify connection
try:
    client.admin.command('ping')
    print("✓ MongoDB connection successful")
except Exception as e:
    print(f"✗ Connection error: {e}")
    exit(1)

# 2. Load the run document
target_run_id = "run_SealQA_2025-12-18T02:41:05.742380"
print(f"Querying by run_id: {target_run_id}")
doc = collection.find_one({"run_id": target_run_id})

if not doc:
    print("Document not found!")
    exit(1)

print(f"Loaded Run: {doc.get('run_id')}")

# 3. Calculate stats from failing_traces
passing_traces = doc.get("passing_traces", [])
failing_traces = doc.get("failing_traces", [])

total = len(passing_traces) + len(failing_traces)
passing = len(passing_traces)
failing = len(failing_traces)

# Count fixed: failing traces with successful_repairs > 0
fixed = 0
analyzed = 0

for trace in failing_traces:
    metrics = trace.get("metrics", {})
    if metrics:
        analyzed += 1
        repairs = metrics.get("repairs", {})
        successful_repairs = repairs.get("successful_repairs", 0)
        if successful_repairs and successful_repairs > 0:
            fixed += 1

# Calculate accuracy: (passing + fixed) / total
accuracy = 0.0
if total > 0:
    accuracy = (passing + fixed) / total

# 4. Display current and new stats
current_stats = doc.get("stats", {})
print("\n=== Current Stats ===")
print(f"Total: {current_stats.get('total', 0)}")
print(f"Passing: {current_stats.get('passing', 0)}")
print(f"Failing: {current_stats.get('failing', 0)}")
print(f"Fixed: {current_stats.get('fixed', 0)}")
print(f"Analyzed: {current_stats.get('analyzed', 0)}")
print(f"Accuracy: {current_stats.get('accuracy', 0.0):.4f}")

print("\n=== Calculated Stats (from traces) ===")
print(f"Total: {total}")
print(f"Passing: {passing}")
print(f"Failing: {failing}")
print(f"Fixed: {fixed} (traces with successful_repairs > 0)")
print(f"Analyzed: {analyzed} (traces with metrics)")
print(f"Accuracy: {accuracy:.4f}")

# 5. Update the document
print("\n=== Update Plan ===")
print(f"Will update stats.fixed: {current_stats.get('fixed', 0)} -> {fixed}")
print(f"Will update stats.analyzed: {current_stats.get('analyzed', 0)} -> {analyzed}")
print(f"Will update stats.accuracy: {current_stats.get('accuracy', 0.0):.4f} -> {accuracy:.4f}")

confirm = input("\nType 'CONFIRM' to apply changes to DB: ")
if confirm == "CONFIRM":
    collection.update_one(
        {"run_id": target_run_id},
        {
            "$set": {
                "stats.fixed": fixed,
                "stats.analyzed": analyzed,
                "stats.accuracy": accuracy,
                # Also update total, passing, failing to match actual counts
                "stats.total": total,
                "stats.passing": passing,
                "stats.failing": failing,
            }
        }
    )
    print("✓ Update successful!")
else:
    print("Operation cancelled.")

client.close()

