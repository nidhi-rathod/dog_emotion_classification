import shutil
from pathlib import Path
import config as cfg

# Pull local directories from our clean config file
DATASET_ROOT = Path(cfg.DATA_RAW_DIR)
UNLABELED_DIR = Path(cfg.UNLABELED_DIR)

# Folders inside the dataset that don't have explicit emotion labels
UNLABELED_FOLDERS = [
    'adult_dog', 'Dogs', 'puppy',
    'Beagle', 'Border Collie', 'Dachshund', 'German Shepherd',
    'Golden Retriever', 'Labrador Retriever', 'Pomeranian',
    'Pug', 'Shih Tzu', 'Siberian Husky'
]

# Ensure the unlabeled target folder exists locally
UNLABELED_DIR.mkdir(parents=True, exist_ok=True)

moved = 0
for folder_name in UNLABELED_FOLDERS:
    src = DATASET_ROOT / folder_name
    
    if not src.exists():
        print(f'Not found, skipping: {folder_name}')
        continue
        
    # Find all sound formats in this folder
    files = list(src.rglob('*.wav')) + list(src.rglob('*.mp3'))
    print(f'{folder_name}: {len(files)} files found')
    
    # Safely copy them over into data/unlabeled/ with a safe unique prefix
    for f in files:
        dest = UNLABELED_DIR / f'unlab_{moved:05d}_{f.name}'
        shutil.copy2(f, dest)
        moved += 1

print(f'\nTotal unlabeled files collected and consolidated: {moved}')