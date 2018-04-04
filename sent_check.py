import pandas as pd

key_List = ['none', 'comp', 'equal', 'hidden', 'most', 'diff']

df = pd.read_excel('./data/jd_comp_final_v5.xlsx')
data_dict = {'none': df[df['Yes/No'] == 0],
              'comp': df[(df['差比'] == 1) & (df['H'] == 0)],
              'most': df[(df['差比'] == 3) & (df['H'] == 0)],
              'diff': df[(df['差比'] == 2) & (df['H'] == 0)],
              'equal': df[(df['平比'] == 1) & (df['H'] == 0)],
              'hidden': df[df['H'] == 1]}

res = {}
for key in key_List:
    dd = data_dict[key]['sent_bin'].value_counts().to_dict()
    shape = data_dict[key].shape[0]
    dd = {k: round(v / shape *100, 2) for k,v in dd.items()}
    res[key] = dd

percent = pd.DataFrame(res).unstack(level=0).unstack(level=-1)
