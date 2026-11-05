from utils import load_script

script1 = load_script("scripts\\script1.txt")
script2 = load_script("scripts\\script2.txt")

scripts = [script1, script2]

print(script1["num_gaps"], script1["budget_K"])
print(script2["num_gaps"], script2["budget_K"])