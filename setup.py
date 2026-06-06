import os

# 1. Create ALL your local folders at once cleanly
for folder in ['data/raw', 'data/processed', 'data/features', 'models', 'outputs']:
    os.makedirs(folder, exist_ok=True)

print("Local folders created successfully!")
print("Action Required: Put your unzipped dataset files inside the 'data/raw' folder now.")

# 2.Check if anything is inside data/raw yet
try:
    print('Current Local dataset contents:', os.listdir('data/raw'))
    print("If the list above is empty, your data isn't in the right place yet!")
except Exception as e:
    print('Error reading folder:', e)