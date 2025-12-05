from pathlib import Path
text = Path('yinzifenxi/fa_nonparam_analysis.py').read_text(encoding='utf-8').splitlines()
for i in range(3300, 3520):
    print(f"{i+1}: {text[i]}")
