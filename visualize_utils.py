import random
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects


def get_colors(n=4096):
    n = round(n**(1/3))
    grid = [(256 // n)*(i+1) - 1 for i in range(16)]
    colors = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                colors.append((grid[i],grid[j],grid[k]))
    random.shuffle(colors)
    return colors

def show_output(img, dets):
    ax = show_img(img, figsize=(16,8))
    for d in dets:
        draw_rect(ax, bbox_plt(d))
        c = classes[d[-1].int().item()]
        draw_text(ax, d[:2], c)


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=0.5))
    draw_outline(patch, 2)


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


def bbox_plt(box):
    bx, by = box[0], box[1]
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    return [bx, by, bw, bh]