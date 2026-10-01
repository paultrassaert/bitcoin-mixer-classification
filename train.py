"""Train on feature CSVs; use internal development data and a separate provided test file."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

EXCLUDED = {'txid', 'label', 'timestamps', 'total_output_btc', 'mean_output_vals',
            'fee_per_input', 'complexity_score', 'log_volume'}
EXPECTED_CLASSES = {'Bitcoin Fog', 'JoinMarket', 'SamouraiWallet.com', 'WasabiWallet.io', 'other'}


def check_data(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    for name, frame in [('train',train),('test',test)]:
        if not {'txid','label'} <= set(frame):
            raise ValueError(f'{name}: require txid and label columns')
        if frame[['txid','label']].isna().any().any() or frame.txid.duplicated().any():
            raise ValueError(f'{name}: missing IDs/labels or duplicate transaction IDs')
        if set(frame.label) != EXPECTED_CLASSES:
            raise ValueError(f'{name}: expected the five documented service classes')
    if set(train.txid) & set(test.txid):
        raise ValueError('Train/test transaction IDs overlap')
    cols = [c for c in train if c not in EXCLUDED]
    if not cols or set(cols) != {c for c in test if c not in EXCLUDED}:
        raise ValueError('Train/test feature schemas differ or are empty')
    for frame in [train,test]:
        if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in cols):
            raise ValueError('Predictors must be numeric; remove metadata explicitly')
        if not np.isfinite(frame[cols].to_numpy()).all():
            raise ValueError('NaN or infinite predictors must be resolved explicitly')
    return cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-features',required=True)
    p.add_argument('--test-features',required=True)
    p.add_argument('--model',choices=['random_forest','xgboost'],default='random_forest')
    p.add_argument('--out',default='results/run')
    p.add_argument('--synthetic',action='store_true',help='Mark demonstration runs as synthetic, never real performance')
    args=p.parse_args()
    train,test=(pd.read_csv(f) for f in [args.train_features,args.test_features])
    cols=check_data(train,test)
    encoder=LabelEncoder().fit(train.label)
    y=encoder.transform(train.label)
    fit_idx,dev_idx=train_test_split(np.arange(len(train)),test_size=.2,random_state=42,stratify=y)
    weights=compute_sample_weight('balanced',y[fit_idx])
    if args.model=='xgboost':
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=800,max_depth=4,learning_rate=.03,
               subsample=.8,colsample_bytree=.8,objective='multi:softprob',num_class=5,
               eval_metric='mlogloss',early_stopping_rounds=30,random_state=42,n_jobs=2)
        model.fit(train.iloc[fit_idx][cols],y[fit_idx],sample_weight=weights,
                  eval_set=[(train.iloc[dev_idx][cols],y[dev_idx])],verbose=False)
    else:
        model=RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=3,
                                     random_state=42,n_jobs=2)
        model.fit(train.iloc[fit_idx][cols],y[fit_idx],sample_weight=weights)
    # Report the provided test file only after all fitting/early stopping is complete.
    truth=encoder.transform(test.label)
    pred=model.predict(test[cols]);prob=model.predict_proba(test[cols])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    result={'status':'SYNTHETIC SMOKE TEST' if args.synthetic else 'provided test set; temporal/entity independence unverified',
            'model':args.model,'n_fit':len(fit_idx),'n_dev':len(dev_idx),'n_test':len(test),
            'classes':encoder.classes_.tolist(), 'features':cols,
            'macro_f1':float(f1_score(truth,pred,average='macro')),
            'macro_ovr_auc':float(roc_auc_score(truth,prob,multi_class='ovr',average='macro')),
            'classification_report':classification_report(truth,pred,target_names=encoder.classes_,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(truth,pred).tolist(),
            'data_sha256':{Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest()
                           for f in [args.train_features,args.test_features]}}
    (out/'test_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','macro_f1','macro_ovr_auc','n_test']},indent=2))


if __name__=='__main__':
    main()
