import json
all_data = json.load(open("data/all.json"))
gen_data = json.load(open("data/generated.json"))
mixed = all_data * 5 + gen_data  # 145 + 324 = 469
json.dump(mixed, open("data/mixed_5x.json", "w"), indent=2)
