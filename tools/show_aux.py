from pathlib import Path
text = Path('yinzifenxi/fa_nonparam_analysis.py').read_text(encoding='utf-8').splitlines()
start = 3135
for i in range(start, start+200):
    print(f"{i+1}: {text[i]}")
