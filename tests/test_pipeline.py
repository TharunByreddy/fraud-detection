
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from src.data_pipeline import generate_synthetic_data,preprocess

def test_shape():
    df=generate_synthetic_data(1000)
    assert len(df)==1000

def test_fraud_rate():
    df=generate_synthetic_data(10000)
    assert 0.01<=df.is_fraud.mean()<=0.05

def test_no_nulls():
    df=generate_synthetic_data(500)
    assert preprocess(df).isnull().sum().sum()==0

def test_log_amount():
    df=generate_synthetic_data(500)
    assert "log_amount" in preprocess(df).columns
