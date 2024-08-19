# _*_ coding:utf-8 _*_
# !/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd
import palettable
import matplotlib.patches as patches
from matplotlib import rc

cpos_x = 0.14004283139013166
cpos_y = 0.060
cpos_h = 0.015
annot_fontsize = 18
ticklabel_size = 27
ytick_labels_pos = 95  # this value is bigger, the ytick_labels is closer to right side. And this value can be set as negative.
tree_pos = 129
tree_ratio = 0.65

data = pd.read_excel(io=r'E:\OSU\In_Progress\Now_work\work\Dm_Heatmap.xlsx',
                     sheet_name=0)  # read data from xlsx file
data1 = data.set_index(['name'])  # set the data of the first column be rownames
data_normalized = np.log2(data1+1)  # parameterize data with log2
maxValues = data_normalized.max().max()
print(maxValues)
print("Data normalization complete!")

# use tex to set text format, like underline
rc('text', usetex=False)

# print(data.columns)

def get_desired_net(age):
    desired_net = {0: "0", 0.50: "1", 1.50: "2", 2.50: "3", 3.50: "4", 4.50: "5", 5.50: "6", 6.50: "7", 7.50: "8",
                   8.50: "9", 9.50: "A", 10.50: "B", 11.50: "C", 12.50: "D", 13.50: "E", 14.50: "F"}

    keys_array = np.array(list(desired_net.keys()))

    max_age = max(keys_array[keys_array <= age])

    return desired_net[max_age]


data_labels = data_normalized.astype(str)

for i in data_normalized.index:
    for j in data_normalized.columns:
        data_labels.at[i, j] = get_desired_net(data_normalized.at[i, j])

# draw preliminary clustermaps
g = sns.clustermap(data_normalized,
                   annot=data_labels, annot_kws={"fontsize": ticklabel_size}, fmt='',
                   figsize=(23 + 0.34 * len(data_normalized.columns), 12 + 0.34 * len(data_normalized)),
                   method='average', metric='euclidean', vmin=0, vmax=10,
                   cbar=False,
                   row_cluster=False, col_cluster=False,
                   # xticklabels=1, yticklabels=1,
                   tree_kws={  # line shape
                       'colors': ['black'] * 10 + ['black'] * 2 + ['black'] * 13,  # line color
                       'linewidths': 1.2},
                   cmap='jet',
                   linewidths=0, linecolor='white'
                   )

g.cax.set_visible(False)
plt.subplots_adjust(left=0, right=0.7, bottom=0.1, top=0.8)

# modify x,y axis of clustermap
ax = g.ax_heatmap  # X&Y axis of heatmap
ax.xaxis.tick_top()  # set x axis to the top
ax.yaxis.tick_right()
ax.set_ylabel("")
ytick_labels = ax.get_yticklabels()
print(ytick_labels)
ax.spines['right'].set_position(('data', 52))
ax.spines['right'].set_visible(False)
ax.set_yticklabels(ytick_labels, ha='left')
ax.tick_params(right=False, top=False, left=False)  # remove axis tick
vline = [0, 12, 18, 24, 27, 30, 34, 36, 39, 40, 42, 46, 47, 48, 50, 52]
hline = [0, 1, 3, 5, 7, 15, 19]
ax.hlines(hline, 0, 500, colors='w')

# set yticks labels color
for i in range(hline[0], hline[1]):
 ax.get_yticklabels()[i].set_color('red')

for i in range(hline[1], hline[2]):
    ax.get_yticklabels()[i].set_color('dodgerblue')

for i in range(hline[2], hline[3]):
    ax.get_yticklabels()[i].set_color('limegreen')

for i in range(hline[3], hline[4]):
    ax.get_yticklabels()[i].set_color('orange')

for i in range(hline[4], hline[5]):
    ax.get_yticklabels()[i].set_color('darkorchid')

for i in range(hline[5], hline[6]):
    ax.get_yticklabels()[i].set_color('saddlebrown')

# set xticks labels color
for i in range(vline[0], vline[1]):
    ax.get_xticklabels()[i].set_color('red')

for i in range(vline[1], vline[2]):
    ax.get_xticklabels()[i].set_color('dodgerblue')

for i in range(vline[2], vline[3]):
    ax.get_xticklabels()[i].set_color('limegreen')

for i in range(vline[3], vline[4]):
    ax.get_xticklabels()[i].set_color('orange')

for i in range(vline[4], vline[5]):
    ax.get_xticklabels()[i].set_color('darkorchid')

for i in range(vline[5], vline[6]):
    ax.get_xticklabels()[i].set_color('violet')

for i in range(vline[6], vline[7]):
    ax.get_xticklabels()[i].set_color('coral')

for i in range(vline[7], vline[8]):
    ax.get_xticklabels()[i].set_color('turquoise')

for i in range(vline[8], vline[9]):
    ax.get_xticklabels()[i].set_color('maroon')

for i in range(vline[9], vline[10]):
    ax.get_xticklabels()[i].set_color('steelblue')

for i in range(vline[10], vline[11]):
    ax.get_xticklabels()[i].set_color('saddlebrown')

for i in range(vline[11], vline[12]):
    ax.get_xticklabels()[i].set_color('mediumslateblue')

for i in range(vline[12], vline[13]):
    ax.get_xticklabels()[i].set_color('pink')

for i in range(vline[13], vline[14]):
    ax.get_xticklabels()[i].set_color('gold')

for i in range(vline[14], vline[15]):
    ax.get_xticklabels()[i].set_color('olivedrab')

plt.draw

plt.setp(ax.get_xticklabels(), size=ticklabel_size, rotation=90)  # set X axis label size and angle
plt.setp(ax.get_yticklabels(), size=ticklabel_size, rotation=360)  # set Y axis label size and angle

box_heatmap = g.ax_heatmap.get_position()

print(box_heatmap)

# set another colorbar
ax3 = g._figure.add_axes([cpos_x, cpos_y, 0.4, cpos_h])  # left, bottom, width, length
norm = mpl.colors.Normalize(vmin=0, vmax=10)
cb3 = mpl.colorbar.ColorbarBase(ax3, cmap='jet',
                                norm=norm,
                                ticklocation='bottom',
                                orientation='horizontal'
                                )

cb3.set_ticks([0.25, 5, 10], labels=["0", "5", "A:10"], fontsize=annot_fontsize)
cb3.outline.set_visible(False)
cb3.ax.tick_params(top=False, bottom=False, labeltop=True, labelbottom=False)
font1 = {'size': annot_fontsize, 'color': '#000000'}  # set font style
cb3.set_label('Log${_2}$(FPKM+1)', fontdict=font1)  # set cbar label and size

# set other axes
ax1 = g._figure.add_axes([box_heatmap.x1+(box_heatmap.x1 - box_heatmap.x0) * 8.9 / vline[-1], box_heatmap.y0, 0, box_heatmap.y1-box_heatmap.y0])
ax1.set_yticks([1/19, 2/19, 3/19, 8/19, 13/19, 15/19, 17/19, 37/38], ['PAP3', 'PAP2', 'PAP1', 'HP8', 'HP6', 'HP5', 'HP21', 'HP14'], fontsize=ticklabel_size)
ax1.yaxis.tick_right()
for i in range(0, 3):
    ax1.get_yticklabels()[i].set_color('saddlebrown')

for i in range(3, 4):
    ax1.get_yticklabels()[i].set_color('darkorchid')

for i in range(4, 5):
    ax1.get_yticklabels()[i].set_color('orange')

for i in range(5, 6):
    ax1.get_yticklabels()[i].set_color('limegreen')

for i in range(6, 7):
    ax1.get_yticklabels()[i].set_color('dodgerblue')

for i in range(7, 8):
    ax1.get_yticklabels()[i].set_color('red')

ax1.tick_params(top=False, bottom=False, left=False, right=False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticklabels('')

ax2 = g._figure.add_axes([box_heatmap.x1+(box_heatmap.x1 - box_heatmap.x0) * 14.3 / vline[-1], box_heatmap.y0, 0, box_heatmap.y1-box_heatmap.y0])
ax2.set_yticks([8/19, 17/19, 37/38], ['cSP23/SPE', 'cSP55/SAE', 'SP137/MSP'], fontsize=ticklabel_size)
ax2.yaxis.tick_right()
for i in range(0, 1):
    ax2.get_yticklabels()[i].set_color('darkorchid')

for i in range(1, 2):
    ax2.get_yticklabels()[i].set_color('dodgerblue')

for i in range(2, 3):
    ax2.get_yticklabels()[i].set_color('red')

ax2.tick_params(top=False, bottom=False, left=False, right=False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_xticklabels('')


# set other figures
lst2 = [(box_heatmap.x1 - box_heatmap.x0) * (vline[1] - vline[0]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[2] - vline[1]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[3] - vline[2]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[5] - vline[3]) / vline[-1],
#        (box_heatmap.x1 - box_heatmap.x0) * (vline[5] - vline[4]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[6] - vline[5]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[7] - vline[6]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[8] - vline[7]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[9] - vline[8]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[10] - vline[9]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[11] - vline[10]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[12] - vline[11]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[13] - vline[12]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[14] - vline[13]) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[15] - vline[14]) / vline[-1],

        (box_heatmap.x1 - box_heatmap.x0) * (vline[5] - vline[0] - 0.3) / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * (vline[15] - vline[5] - 0.3) / vline[-1],

        (box_heatmap.x1 - box_heatmap.x0) * 7.55 / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * 4 / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * 5 / vline[-1],
        (box_heatmap.x1 - box_heatmap.x0) * 11 / vline[-1]
        ] + [0.00125]*9

lst1 = [box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[0] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[1] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[2] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[3] / vline[-1],
#        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[4] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[5] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[6] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[7] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[8] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[9] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[10] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[11] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[12] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[13] / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * vline[14] / vline[-1],

        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[0]+0.3) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[5]+0.3) / vline[-1],

        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+8.9) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+14) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+9) / vline[-1],

        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+8.6) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+8.6) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+8.6) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+8.6) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+8.6) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+8.6) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+14) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+14) / vline[-1],
        box_heatmap.x0 + (box_heatmap.x1 - box_heatmap.x0) * (vline[15]+14) / vline[-1]
        ]

#group parameter
name = ['egg', 'larva', 'pupa', 'adult', 'gut', '', 'fat body', '', '', 'carcas', '', '', 'ovary', '', 'whole body',
        'body parts', '$\it{D. melanogaster}$', '$\it{M. sexta}$', '$\it{T. molitor}$', 'orthologous relationships',
        '', '', '', '', '', '', '', '', '']
color = ['red']*4 + ['dodgerblue']*10 + ['black']*6 + ['white']*9
line_color = ['white']*14 + ['red'] + ['dodgerblue'] + ['white']*3 + ['black'] + ['red', 'dodgerblue', 'limegreen',
                                                                                  'orange', 'darkorchid', 'saddlebrown',
                                                                                  'red', 'dodgerblue', 'darkorchid']
line_width = [0.0001]*14 + [0.0025]*2 + [0.0001]*3 + [0.0005] + [(box_heatmap.y1 - box_heatmap.y0) * (hline[1]-hline[0]-0.2) / hline[-1],
        (box_heatmap.y1 - box_heatmap.y0) * (hline[2]-hline[1]-0.2) / hline[-1],
        (box_heatmap.y1 - box_heatmap.y0) * (hline[3]-hline[2]-0.2) / hline[-1],
        (box_heatmap.y1 - box_heatmap.y0) * (hline[4]-hline[3]-0.2) / hline[-1],
        (box_heatmap.y1 - box_heatmap.y0) * (hline[5]-hline[4]-0.2) / hline[-1],
        (box_heatmap.y1 - box_heatmap.y0) * (hline[6]-hline[5]-0.2) / hline[-1],
        (box_heatmap.y1 - box_heatmap.y0) * (hline[1]-hline[0]-0.2) / hline[-1],
        (box_heatmap.y1 - box_heatmap.y0) * (hline[2]-hline[1]-0.2) / hline[-1],
        (box_heatmap.y1 - box_heatmap.y0) * (hline[5]-hline[4]-0.2) / hline[-1]]
tick_location = ['bottom']*19 + ['top'] + ['left']*9
Orientation = ['horizontal']*20 + ['vertical']*9
xgroup_label = [(box_heatmap.y1 - box_heatmap.y0) * 5.2 / hline[-1]]*14\
               + [(box_heatmap.y1 - box_heatmap.y0) * 6.2 / hline[-1]]*2\
               + [0.03]*3 + [0.042]\
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[1]-0.1) / hline[-1]] \
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[2]-0.1) / hline[-1]]\
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[3]-0.1) / hline[-1]]\
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[4]-0.1) / hline[-1]]\
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[5]-0.1) / hline[-1]]\
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[6]-0.1) / hline[-1]]\
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[1]-0.1) / hline[-1]]\
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[2]-0.1) / hline[-1]]\
               + [(box_heatmap.y0 - box_heatmap.y1) * (hline[5]-0.1) / hline[-1]]


def label_xticktables(lst1, lst2, name, color, xgrouplabel, tick_location, Orientation, line_color, line_width):
    cmap = mpl.colors.ListedColormap([line_color])
    ax4 = g._figure.add_axes([lst1, box_heatmap.y1 + xgrouplabel, lst2, line_width])
    cb4 = mpl.colorbar.ColorbarBase(ax4, cmap=cmap,
    ticklocation=tick_location,
    orientation=Orientation
    )
    cb4.outline.set_visible(False)
    cb4.ax.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False)
    font_label = {'size': ticklabel_size, 'color': color}
    cb4.set_label(name, fontdict=font_label)


for z in range(len(name)):
    label_xticktables(lst1[z], lst2[z], name[z], color[z], xgroup_label[z], tick_location[z], Orientation[z], line_color[z], line_width[z])


import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D, offset_copy


def rainbow_text(x, y, strings, colors, orientation='horizontal', ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    fig = ax.figure
    canvas = fig.canvas

    rc('text', usetex=True)

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        # Convert window extent from pixels to inches
        # to avoid issues displaying at different dpi
        ex = fig.dpi_scale_trans.inverted().transform_bbox(ex)

        if orientation == 'horizontal':
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=ex.width, y=0)
        else:
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=0, y=ex.height)

plt.savefig('Dm_Heatmap.pdf', dpi=400)

print("Draw clustermap successfully!")
