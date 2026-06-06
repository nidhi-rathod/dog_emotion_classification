import sys
import os
import logging
import numpy as np
from pathlib import Path
import config as cfg
from utils.features import extract_features

# Set up clean logging structure
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_feature_extraction():
    src_root = Path(cfg.DATA_PROC_DIR)
    dst_root = Path(cfg.FEATURES_DIR)
    os.makedirs(dst_root, exist_ok=True)
    
    if not src_root.exists(): 
        print('Error: Run preprocess_data.py first to generate processed files!')
        return
        
    label_map = {c: i for i, c in enumerate(cfg.EMOTION_CLASSES)}
    print(f'Label mapping: {label_map}\n')
    
    X, y_out, errors = [], [], 0
    
    for label, label_idx in label_map.items():
        ep = src_root / label
        if not ep.exists(): 
            logger.warning(f'Skipping {label}')
            continue
            
        files = list(ep.glob('*.wav'))
        logger.info(f'[{label}] Extracting features for {len(files)} files...')
        
        for f in files:
            try:
                feat = extract_features(str(f))
                X.append(feat)
                y_out.append(label_idx)
                
                # SpecAugment: frequency mask (zeros out random horizontal lines)
                feat_fmask = feat.copy()
                f0 = np.random.randint(0, feat.shape[0] // 4)
                f_start = np.random.randint(0, feat.shape[0] - f0 + 1)
                feat_fmask[f_start:f_start + f0, :] = 0.0
                X.append(feat_fmask)
                y_out.append(label_idx)
                
                # SpecAugment: time mask (zeros out random vertical slices)
                feat_tmask = feat.copy()
                t0 = np.random.randint(0, feat.shape[1] // 4)
                t_start = np.random.randint(0, feat.shape[1] - t0 + 1)
                feat_tmask[:, t_start:t_start + t0] = 0.0
                X.append(feat_tmask)
                y_out.append(label_idx)
                
            except Exception as e:
                print(f'Skip {f.name} due to feature error: {e}')
                errors += 1
                
    X = np.array(X)
    y_out = np.array(y_out)
    
    # Save arrays locally as binary NumPy dumps
    np.save(dst_root / 'X_features.npy', X)
    np.save(dst_root / 'y_labels.npy', y_out)
    
    print(f'\nPipeline 2 complete. Saved {len(X)} samples to {dst_root}.')
    print(f'Errors encountered: {errors}')
    for i, cls in enumerate(cfg.EMOTION_CLASSES):
        print(f'  {cls:<14}: {np.sum(y_out == i)} samples')

# ── LOCAL RESUME GUARD ──
if __name__ == "__main__":
    X_path = os.path.join(cfg.FEATURES_DIR, 'X_features.npy')
    y_path = os.path.join(cfg.FEATURES_DIR, 'y_labels.npy')
    
    if os.path.exists(X_path) and os.path.exists(y_path):
        print('Pipeline 2 already complete — skipping feature extraction.')
        print('To force a full rerun, delete these files from your "data/features/" directory.')
        X = np.load(X_path)
        y = np.load(y_path)
        print(f'Loaded existing processed matrices: X={X.shape} y={y.shape}')
    else:
        run_feature_extraction()