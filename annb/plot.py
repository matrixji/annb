from typing import List
import matplotlib.pyplot as plt

from .result import BenchmarkResult


def plot_result_recall_vs_qps(results: List[BenchmarkResult], **kwargs):
    output = kwargs.get('output', 'result.png')
    if not output:
        output = 'result.png'
    if not output.endswith('.png'):
        output += '.png'

    plt.style.use('seaborn-whitegrid')
    # create figure from results
    fig, ax = plt.subplots()

    title = kwargs.get('title', 'Recall vs QPS')
    subtitle = kwargs.get('subtitle', '')
    if subtitle:
        title += '\n' + subtitle
    ax.set_title(title)

    ax.set_xlabel('Recall')
    ax.set_ylabel('QPS')
    ax.set_xlim(0.0, 1.0)

    lines = []
    labels = []
    for result in results:
        recalls = [r.recall for r in result.query_results]
        qps = [BenchmarkResult.qps(r.durations) for r in result.query_results]
        lines.append(
            ax.plot(
                recalls, qps, linestyle='-', marker='o', label=result.attributes['name']
            )
        )
        labels.append(result.attributes['name'])
    foot_notes = kwargs.get('foot_notes', '')
    if foot_notes:
        ax.text(
            0.0,
            -0.1,
            foot_notes,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
        )
    fig.legend(loc='right', bbox_to_anchor=(1.2, 0.5), ncol=1)
    fig.savefig(output, bbox_inches='tight', dpi=300)
