import pandas as pd
import argparse
from pathlib import Path
import plotly.graph_objects as go
import random


def get_random_hex_color():
    s = "#"
    
    for _ in  range(6):
        s += random.choice("0123456789ABCDEF")
    return s


def generate_hex_colors(n_colors=10):
    colors = []
    for i in range(n_colors):
        colors.append(get_random_hex_color())
    return colors


def parse_args():
    parser = argparse.ArgumentParser(description='search ')
    parser.add_argument('--metrics_folder', type=str, default='results/dataset_metrics/', help='path to the folder with metrics csv files')
    parser.add_argument('--save_path', type=str, default='results/dataset_metrics_hists', help='results save path')
    parser.add_argument('--metric', type=str, default='', help='which metric to plot, separate by commas')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    random.seed(28)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    if args.metric:
        sorting_criterion = args.metric.split(',')[0]
    else:
        sorting_criterion = 'R@1'
    
    metrics_folder = Path(args.metrics_folder)
    metrics_csv_paths = list(metrics_folder.glob('*.csv'))
    colors = None


    for metrics_csv_path in metrics_csv_paths:
        dataset_name = metrics_csv_path.stem

        df = pd.read_csv(
            metrics_csv_path, 
            index_col=0
        )
        df = df.sort_values(
            by=sorting_criterion, 
            ascending = False
        )
        df = df.transpose() 

        model_names = list(df.columns)
        df['metric'] = df.index

        if colors is None:
            colors_hex = generate_hex_colors(len(model_names))
            colors = {model_names[idx]:colors_hex[idx]for idx in range(len(model_names))}

        if args.metric:
            metrics = args.metric.split(',')
        else:
            metrics = df['metric']

        x = list(range(len(metrics)))

        bar_plots = []

        for model_name in model_names:
            bar_plots.append(go.Bar(
                x=x, 
                y=df[model_name], 
                name=model_name, 
                marker=go.bar.Marker(color=colors[model_name])
                )
            )

        layout = go.Layout(
            title=go.layout.Title(text=f"{dataset_name}", x=0.5),
            yaxis_title="Metric value",
            xaxis_tickmode="array",
            xaxis_tickvals=list(range(27)),
            xaxis_ticktext=tuple(df['metric'].values),
        )

        fig = go.Figure(data=bar_plots, layout=layout)
        fig.write_html(save_path / f"{dataset_name}.html")
    
    
        
    
    
