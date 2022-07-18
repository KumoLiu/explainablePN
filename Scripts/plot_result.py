#%%
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd

prob_path = Path("/homes/yliu/Data/pn_cls_exp/lidc-82-N-new/0707_1741-hesam_cbam-2head-slice_5-bs_40-lr_0.01-plateau-DWA-BCE-sgd-mean-parallel-cbam-mask-RM1-featworound/Models/0712_1421-BestModel@72with20.285.pt/prob_all1.csv")
df = pd.read_csv(prob_path, index_col=None)

fig = go.Figure()

fig.add_trace(go.Violin(x=df['cohort'][ df['label'] == 0 ],
                        y=df['Malignance'][ df['label'] == 0 ],
                        legendgroup='Benign', scalegroup='Benign', name='Benign',
                        side='negative',
                        line_color='blue',
                        width=1)
             )
fig.add_trace(go.Violin(x=df['cohort'][ df['label'] == 1 ],
                        y=df['Malignance'][ df['label'] == 1 ],
                        legendgroup='Malignance', scalegroup='Malignance', name='Malignance',
                        side='positive',
                        line_color='orange',
                        width=1)
             )
fig.update_traces(meanline_visible=True)
fig.update_layout(title_text='', violingap=0, violinmode='overlay',
                  height=800, width=1200,  # 设置图大小
                  font=dict(size=18),  # 字体大小
                  legend={'font': dict(size=25),  # legend字体大小
                          'itemsizing': 'constant', },  # legend图标大小
                  hoverlabel=dict(font=dict(size=20)))
# fig.show()
fig.write_image(prob_path.parent/'violin1.pdf')

#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

prob_path = Path("/homes/yliu/Data/pn_cls_exp/lidc-82-N-new/0707_1741-hesam_cbam-2head-slice_5-bs_40-lr_0.01-plateau-DWA-BCE-sgd-mean-parallel-cbam-mask-RM1-featworound/Models/0712_1421-BestModel@72with20.285.pt/prob_all1.csv")
df = pd.read_csv(prob_path, index_col=None)

plt.figure(dpi=150)

# sns.violinplot(x="cohort", y="Malignance", hue="label",
#                data=df, palette="muted", split=True, cut=True)

sns.violinplot(data=df, x="cohort", y="Malignance", hue="label",
               split=True, palette="Set2",
               scale="count", inner="quartile",
               scale_hue=True)
# sns.despine(left=True)

plt.savefig(prob_path.parent/'violin2.jpg')
# %%
