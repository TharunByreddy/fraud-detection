
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def generate_synthetic_data(n_samples=100000):
    np.random.seed(42)
    nf=int(n_samples*0.017)
    nl=n_samples-nf
    def make(n,fraud):
        return pd.DataFrame({"amount":np.random.lognormal(3.5 if fraud else 4.2,1.2,n).clip(1,25000),
            "hour":np.random.randint(0,24,n),"day_of_week":np.random.randint(0,7,n),
            "v1":np.random.normal(-3 if fraud else 0,2,n),"v2":np.random.normal(2.5 if fraud else 0,2,n),
            "v3":np.random.normal(-4 if fraud else 0,2,n),"v4":np.random.normal(3 if fraud else 0,1.5,n),
            "card_type":np.random.choice(["visa","mastercard","amex"],n),
            "merchant_category":np.random.choice(["retail","travel","food","online","atm"],n),
            "is_fraud":int(fraud)})
    df=pd.concat([make(nf,True),make(nl,False)]).sample(frac=1,random_state=42).reset_index(drop=True)
    print(f"Rows:{len(df):,} Fraud:{df.is_fraud.mean()*100:.2f}%")
    return df

def preprocess(df):
    df=df.copy()
    for c in ["card_type","merchant_category"]:
        df[c]=LabelEncoder().fit_transform(df[c])
    df["log_amount"]=np.log1p(df["amount"])
    return df.drop(columns=["amount"])

def split_and_balance(df):
    X,y=df.drop(columns=["is_fraud"]),df["is_fraud"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    Xtr,ytr=SMOTE(random_state=42).fit_resample(Xtr,ytr)
    sc=StandardScaler()
    Xtr=pd.DataFrame(sc.fit_transform(Xtr),columns=X.columns)
    Xte=pd.DataFrame(sc.transform(Xte),columns=X.columns)
    return Xtr,Xte,ytr,yte,sc
