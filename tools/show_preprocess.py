from pathlib import Path
text = Path('yinzifenxi/fa_nonparam_analysis.py').read_text(encoding='utf-8').splitlines()
start = 719
for i in range(start, start+80):
    print(f"{i+1}: {text[i]}")
