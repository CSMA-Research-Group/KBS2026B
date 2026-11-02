#!/usr/bin/env python3
"""Reproduce the two current figures from saved inputs, writing a new output directory."""
import csv
import json
import os
from pathlib import Path
from workspace import ROOT, current_batch, io_resources, new_run, now, read_json, sha256, write_json

def schematic(out):
    """Vector schematic of two point-agent reachability cases, not new data."""
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, opened in zip(axes, [False, True]):
        for x in range(6):
            for y in range(4):
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor='#f3f5f7', edgecolor='white'))
        wall = [(3, y) for y in range(4) if not (opened and y == 2)]
        for x, y in wall:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor='#48556a'))
        ax.scatter([1.5, 4.5], [1.5, 2.5], marker='^', s=180, color=['#1774a8', '#da8131'], zorder=3)
        ax.scatter([0.5], [1.5], marker='s', s=90, color='#309474', zorder=3)
        for x, y, label in [(1.5, 1.5, 'a'), (4.5, 2.5, 'b'), (0.5, 1.5, 'target')]:
            ax.text(x, y - .43, label, ha='center', va='top', fontsize=10)
        ax.set(xlim=(0, 6), ylim=(0, 4), aspect='equal')
        ax.set_title('Passage open: two feasible goals' if opened else 'Wall closed: one feasible goal', fontsize=11)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'running_scene.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)


def main():
    out = new_run('figures')
    write_json(out / 'resources.json', io_resources())
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    import numpy as np
    batch, _ = current_batch()
    files = list((batch / 'cpu').glob('analysis_*/costs_and_decisions.csv'))
    if len(files) != 1:
        raise ValueError('Ambiguous saved CPU analysis')
    with files[0].open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['split'] == 'test']
    methods = list(dict.fromkeys(r['method'] for r in rows))
    points = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, count in zip(axes.flat, [16, 64, 256, 1024]):
        for method in methods:
            values = sorted(int(r['cost_proxy']) for r in rows if int(r['candidate_count']) == count and r['method'] == method)
            if len(values) != 512:
                raise ValueError('Incomplete CPU plotting group')
            ys = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, ys, where='post', label=method)
            points.append({'candidate_count': count, 'method': method, 'x': values, 'y': ys.tolist()})
        ax.set(title=f'{count} candidates; 512 layouts', xlabel='Audited operation-count proxy',
               ylabel='Correct decision fraction', xscale='log')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=7)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(out / 'cost_to_decision.pdf', metadata={'CreationDate': None})
    plt.close(fig)
    schematic(out)
    if points != read_json(ROOT / 'data/illustrations/cpu_plot_points.json'):
        raise ValueError('Plotting points changed')
    hashes = {p.name:sha256(p) for p in out.glob('*.pdf')}
    original = read_json(ROOT / 'data/illustrations/receipt.json')['figures']
    write_json(out / 'cpu_plot_points.json', points)
    state = {'status':'PASS','at':now(),'directory':str(out.relative_to(ROOT)),
             'source_sha256':sha256(files[0]),'points_equal':True,'points':sum(len(p['x']) for p in points),
             'figures':hashes,'pdf_bytes_equal':hashes == original,'new_experimental_observations':0,
             'note':'PDF bytes may differ with the rendering environment; the input points must be identical.'}
    write_json(out / 'receipt.json', state)
    write_json(ROOT / 'work/latest_figures.json', state)
    print(json.dumps(state))


if __name__ == '__main__':
    main()
