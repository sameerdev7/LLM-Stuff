import re 
from collections import Counter 

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()


# tokenize into words 
tokens = re.findall(r"[a-z']+", text)
print(f"Total tokens: {len(tokens)}")
print("Sample: ", tokens[:30])