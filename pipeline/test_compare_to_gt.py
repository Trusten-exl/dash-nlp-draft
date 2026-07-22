"""
Plain-assert check for the GT-string parsing / scoring helpers in
compare_to_gt.py. Run: python test_compare_to_gt.py
"""

from compare_to_gt import parse_gt_entities, _norm

assert parse_gt_entities("Nvidia:company;Tim Cook:executive") == {
    "nvidia": "company",
    "tim cook": "executive",
}
assert parse_gt_entities("") == {}
assert parse_gt_entities(float("nan")) == {}

assert _norm("  Report ") == "report"
assert _norm(None) == ""
assert _norm(float("nan")) == ""

print("test_compare_to_gt: all checks passed")
