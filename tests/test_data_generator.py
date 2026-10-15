from src.data.generate_data import generate


def test_generator_is_reproducible():
    clients_a, tx_a = generate(seed=7, clients=20, transactions=100)
    clients_b, tx_b = generate(seed=7, clients=20, transactions=100)
    assert clients_a.equals(clients_b)
    assert tx_a.equals(tx_b)


def test_generator_has_valid_labels_and_client_links():
    clients, tx = generate(seed=9, clients=20, transactions=100)
    assert set(tx["is_synthetic_fraud"].unique()).issubset({0, 1})
    assert tx["client_id"].isin(clients["client_id"]).all()
    assert tx["transaction_id"].is_unique
