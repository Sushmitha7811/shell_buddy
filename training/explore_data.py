import random
from collections import Counter
from datasets import load_dataset

ds = load_dataset("AnishJoshi/nl2bash-custom")
print(ds)   # shows the splits and row counts

train = ds["train"]
random.seed(0)
print("\n=== 15 random examples ===")
for i in random.sample(range(len(train)), 15):
    row = train[i]
    print(f"\nNL:   {row['nl_command']}\nBASH: {row['bash_code']}")

cmds = train["bash_code"]
first_words = Counter(c.split()[0] for c in cmds if c.split())
print("\n=== 20 most common programs ===")
print(first_words.most_common(20))

lengths = [len(c) for c in cmds]
print(f"\nCommand length (chars): min {min(lengths)}, max {max(lengths)}, avg {sum(lengths) / len(lengths):.0f}")
print("Multi-line commands:", sum("\n" in c for c in cmds))
print("Duplicate commands:", len(cmds) - len(set(cmds)))
