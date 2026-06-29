
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def tune_threshold(y_true,y_prob):
    p,r,t=precision_recall_curve(y_true,y_prob)
    f1=2*p*r/(p+r+1e-8)
    idx=np.argmax(f1)
    best=float(t[idx]) if idx<len(t) else 0.5
    print(f"Threshold:{best:.4f} F1:{f1[idx]:.4f}")
    return best

def evaluate(name,y_true,y_prob,thr=0.5):
    y_pred=(y_prob>=thr).astype(int)
    print(f"[{name}] ROC={roc_auc_score(y_true,y_prob):.4f} PR={average_precision_score(y_true,y_prob):.4f} F1={f1_score(y_true,y_pred):.4f}")
    return {"model":name,"ROC-AUC":roc_auc_score(y_true,y_prob),"PR-AUC":average_precision_score(y_true,y_prob),"F1":f1_score(y_true,y_pred)}

def train_xgboost(Xtr,ytr,Xte,yte):
    m=XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=6,
        scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(),random_state=42,n_jobs=-1)
    m.fit(Xtr,ytr,eval_set=[(Xte,yte)],verbose=False)
    prob=m.predict_proba(Xte)[:,1]
    thr=tune_threshold(yte,prob)
    joblib.dump({"model":m,"threshold":thr},"models/xgboost.pkl")
    return evaluate("XGBoost",yte,prob,thr)
