"""
Plain-assert check that batching full_article_sent/ent_sent produces the same
aggregation as the old one-call-per-sentence loop. Run: python test_sentiment.py
"""

import ast


class _FakeSent:
    def __init__(self, text):
        self.text = text


class _FakeDoc:
    def __init__(self, texts):
        self.sents = [_FakeSent(t) for t in texts]


src = open('sentiment.py').read()
tree = ast.parse(src)
ns = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('full_article_sent', 'ent_sent'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<sentiment-core>', 'exec'), ns)

full_article_sent = ns['full_article_sent']
ent_sent = ns['ent_sent']


# --- full_article_sent: one call, 3 sentences, one of each label ---
def stub_full_sentiment(texts):
    labels = {"good": "positive", "bad": "negative", "meh": "neutral"}
    return [{"label": labels[t], "score": 0.9} for t in texts]


ns['full_sentiment'] = stub_full_sentiment
doc = _FakeDoc(["good", "bad", "meh"])
article_sent, sent_sent = full_article_sent(doc)

assert article_sent["positive sentences"] == 1
assert article_sent["negative sentences"] == 1
assert article_sent["neutral sentences"] == 1
assert abs(article_sent["score"] - ((0.9 + -0.9 + 0) / 3)) < 1e-9
assert [s["sentiment"] for s in sent_sent] == ["positive", "negative", "neutral"]

# empty article doesn't crash
ns['full_sentiment'] = lambda texts: []
empty_sent, empty_list = full_article_sent(_FakeDoc([]))
assert empty_sent["score"] == 0.0 and empty_list == []


# --- ent_sent: 2 entities, one mentioned twice, one once ---
def stub_ent_sentiment(inputs):
    # inputs: list of {"text": sentence, "text_pair": entity}
    out = []
    for item in inputs:
        if item["text_pair"] == "Nvidia":
            out.append({"label": "Positive", "score": 0.8})
        else:
            out.append({"label": "Negative", "score": 0.7})
    return out


ns['ent_sentiment'] = stub_ent_sentiment
doc = _FakeDoc(["s0", "s1", "s2"])
entities = {
    "Nvidia": {"sentences": [0, 1]},
    "Intel": {"sentences": [2]},
}
result = ent_sent(entities, doc)

assert result["Nvidia"] == (0.8, 2)
assert result["Intel"] == (-0.7, 1)
assert list(result.keys()) == ["Nvidia", "Intel"]  # insertion order preserved

print("test_sentiment: all checks passed")
