from pathlib import Path
text = Path('yinzifenxi/yinzifenxi_main.py').read_text(encoding='utf-8').splitlines()
for idx,line in enumerate(text,1):
    if 'Factor analysis core computation' in line or 'Auxiliary robustness statistics merged' in line or 'Summary dataframe generated' in line or 'Main single-factor report generated' in line:
        print(idx, line.strip())
