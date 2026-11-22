# ./reset_chroma.py
# this script is to reset the vector db
# consider to run it before running transformer script

import shutil
import os

# path to vector directory
db_path = "./chroma_db"

if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("[INFO] Old Chroma DB directory deleted!")
else:
    print("[ERROR] No Chroma db directory found.")

# Recreate an empty directory
os.makedirs(db_path)
print("[WARNING] Chroma db directory is ready to create the vector.")
