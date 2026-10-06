import sqlite3

from svcr.conceptnet import CommonsenseEnricher, ConceptNetClient, LocalConceptNet
from svcr.config import ConceptNetConfig


def test_client_back_off(toy_client):
    concept, edges = toy_client.resolve("a firm handshake")
    assert concept == "handshake" and edges
    concept, edges = toy_client.resolve("Shaking hands")
    assert edges, "multi-word phrase should resolve (shake_hands or its head)"


def test_enrichment_supports_greeting(toy_client, taxonomy, handshake_visual, handshake_draft):
    res = CommonsenseEnricher(toy_client, taxonomy).enrich(handshake_visual, handshake_draft)
    act = res.support["activity"]
    assert max(act, key=act.get) == "greeting"
    assert act["greeting"] > act["eating_together"]
    assert act["greeting"] > act["fighting"]
    assert res.disambiguated["activity"] == "greeting"
    assert res.evidence["activity"]["greeting"], "evidence paths must be reported"
    assert any(f.end == "greeting" or f.start == "greeting" for f in res.facts)
    rel = res.support["relationship"]
    assert rel["colleagues"] > rel["parent_child"]


def test_local_sqlite_backend(tmp_path):
    db = tmp_path / "cn.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE edges (rel TEXT, start TEXT, end TEXT, weight REAL, surface TEXT)")
    conn.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", [
        ("UsedFor", "handshake", "greeting", 3.0, ""), ("RelatedTo", "hug", "handshake", 1.0, ""),
    ])
    conn.commit()
    conn.close()
    client = ConceptNetClient(LocalConceptNet(db), ConceptNetConfig())
    edges = client.edges("handshake")
    assert {(e.start, e.end) for e in edges} == {("handshake", "greeting"), ("hug", "handshake")}
    assert edges[0].weight == 3.0
