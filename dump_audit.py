import json

from data_prep import load_and_audit

df, audit = load_and_audit()
out = {k: v for k, v in audit.items() if k != "products"}
print(json.dumps(out, indent=2, default=str))
with open("audit_facts.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("products:", audit["product_count"], audit["products"])
print("unmapped:", audit["unmapped_products"], "fallback:", audit["fallback_products"])