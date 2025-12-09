# -*- coding: utf-8 -*-
from pathlib import Path
path = Path('yinzifenxi/fa_param_report.py')
text = path.read_text(encoding='utf-8')
old_cols = "    ranking_columns = ['����', '��������', '��������', '�ۺϵ÷�', '����ƽ����÷�', '�껯������', '�껯���ձ���', 'ƽ��ÿ��������', '���س�', '����������', '��������']"
new_cols = "    ranking_columns = ['����', '��������', '��������', '�ۺϵ÷�', '����ƽ����÷�', '�껯������', '�껯���ձ���', 'ƽ��ÿ��������', '���س�', '数据年份', '����������', '��������']"
old_headers = "    ranking_headers = ['����', '����', '��������', '�ۺϵ÷�', '����ƽ����÷�', '�껯������', '�껯����', 'ƽ��ÿ�ʽ���������', '���س�', '����������', '��������']"
new_headers = "    ranking_headers = ['����', '����', '��������', '�ۺϵ÷�', '����ƽ����÷�', '�껯������', '�껯����', 'ƽ��ÿ�ʽ���������', '���س�', '数据年份', '����������', '��������']"
old_table = "        ranking_df = ordered_df[['��������', '��������', '�ۺϵ÷�', '����ƽ����÷�', '�껯������', '�껯���ձ���', 'ƽ��ÿ��������', '���س�', '����������', '��������']].copy()"
new_table = "        ranking_df = ordered_df[['��������', '��������', '�ۺϵ÷�', '����ƽ����÷�', '�껯������', '�껯���ձ���', 'ƽ��ÿ��������', '���س�', '数据年份', '����������', '��������']].copy()"
old_formatter = "        '���س�': lambda x: _fmt_percent(x, 1),\n        '����������': lambda x: str(int(x)) if pd.notna(x) else \"--\","
new_formatter = "        '���س�': lambda x: _fmt_percent(x, 1),\n        '数据年份': lambda x: x if pd.notna(x) and str(x).strip() else \"--\",\n        '����������': lambda x: str(int(x)) if pd.notna(x) else \"--\","
if old_cols not in text or old_headers not in text or old_table not in text or old_formatter not in text:
    raise SystemExit('original text not found')
text = text.replace(old_cols, new_cols, 1)
text = text.replace(old_headers, new_headers, 1)
text = text.replace(old_table, new_table, 1)
text = text.replace(old_formatter, new_formatter, 1)
path.write_text(text, encoding='utf-8')
