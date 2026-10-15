from src.data.generate_data import generate
from src.tools import financial
def test_summary_and_aggregate(tmp_path,monkeypatch):
    c,t=generate(seed=11,clients=20,transactions=200); p=tmp_path/"generated"; p.mkdir()
    c.to_csv(p/"clients.csv",index=False); t.to_csv(p/"transactions.csv",index=False)
    monkeypatch.setattr(financial,"DATA",p); cid=c.iloc[0].client_id
    s=financial.client_summary(cid); a=financial.transaction_aggregate(cid)
    assert a["transaction_count"]==s["transaction_count"]
    assert a["total_amount"]==s["total_transaction_amount"]
