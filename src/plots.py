import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns

def draw_plots(df, column, plot_type : str = "plot"):
    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(12, 8))

    if plot_type == "plot":
        ax.plot(df.index, df[column])

        # Define the locator as before to control tick frequency
        locator = mdates.MonthLocator() # Explicitly locate ticks at the start of each month
        
        # Replace ConciseDateFormatter with the explicit DateFormatter
        formatter = mdates.DateFormatter('%b %Y') # Format: Abbreviated Month and Full Year

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        fig.autofmt_xdate(rotation=45) # rotation for better spacing

    plt.show()

def draw_plots_grid(df, columns,  windows=[7, 28],figure_size = (20,16),save_path = "",nrows = 2, ncols = 2, title_fontsize=20, label_fontsize=16, tick_fontsize=14, legend_fontsize=14):

    df.index = pd.to_datetime(df.index)

    plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'savefig.transparent': True,    # or True if you want to overlay on colored slides
    })


    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figure_size)
    axes_flat = ax.flatten()

    orig_color = '#0070C0'
    rolling_colors = ['#ED7D31', '#229B5A']
    for i, col_name in enumerate(columns):
        current_ax = axes_flat[i]
        current_ax.plot(df.index, df[col_name], label="original", color = orig_color, linewidth = 1)

        for j, window_size in enumerate(windows):
            rolling_mean = df[col_name].rolling(window=window_size).mean()
            current_ax.plot(df.index, rolling_mean, label=f'{window_size}-Day Avg', color=rolling_colors[j], linewidth=2 + j)

        # Set a title for the individual subplot.
        current_ax.set_title(f"Time Series of {col_name}", fontsize=title_fontsize)

        # Define locator and formatter for THIS specific subplot's x-axis.
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter('%b %Y')
        current_ax.xaxis.set_major_locator(locator)
        current_ax.xaxis.set_major_formatter(formatter)

        # Set font sizes for labels and ticks
        current_ax.tick_params(axis='both', labelsize=tick_fontsize)
        current_ax.set_xlabel("Date", fontsize=label_fontsize)
        current_ax.set_ylabel(col_name, fontsize=label_fontsize)

        current_ax.tick_params(axis='x', rotation=45)

        current_ax.legend(fontsize=legend_fontsize)
        # current_ax.grid(True, linestyle='--', alpha=0.6)
        current_ax.grid(color='#DDDDDD', linestyle='-', linewidth=0.8)
        # remove top/right spines
        for spine in ['top','right']:
            current_ax.spines[spine].set_visible(False)
        # optionally thicken bottom/left
        for spine in ['left','bottom']:
            current_ax.spines[spine].set_linewidth(1.2)

    fig.tight_layout(pad=3.0)
    for i in range(len(columns), len(axes_flat)):
        axes_flat[i].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_missing_values_heatmap(df, save_path= ""):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.isnull(), cbar = False, yticklabels = False, cmap = "viridis")
    plt.title('Heatmap of Missing Values by Time', fontsize=16)
    plt.xlabel('Features')
    plt.ylabel('Time (Chronological Order)')
    if(save_path):
        plt.savefig(f"./figures/{save_path}")
    plt.show()

def plot_missing_values_correlation(df, save_path= ""):
    numeric_columns = df.select_dtypes(include = np.number).columns
    plt.figure(figsize=(18, 15))
    sns.heatmap(df[numeric_columns].isnull().corr(), cmap='coolwarm')
    plt.title('Correlation Matrix of Missing Values')
    if(save_path):
        plt.savefig(f"./figures/{save_path}")
    plt.show()


def plot_distribution(df, columns, type = "hist",nrows = 2, ncols = 2, save_path = "", figure_size = (15,12),title_fontsize=20, label_fontsize=16, tick_fontsize=14, legend_fontsize=14):

    plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'savefig.transparent': True,    # or True if you want to overlay on colored slides
    })

    if(type == "heatmap"):
        plt.figure(figsize= figure_size)
        sns.heatmap(
                    df[columns].corr(),
                    cmap="coolwarm",
                    annot=True,
                    fmt=".2f",
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 12}
                )
        plt.title(f"Correlation Heatmap", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(fontsize=12)
        if(save_path):
            plt.savefig(f"./figures/{save_path}.png", dpi = 300)
        plt.show()
        return
    

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    axes_flat = ax.flatten()


    for i, column in enumerate(columns):
        if(type == "heatmap"):
            continue
        current_ax = axes_flat[i]

        if(type == "hist"):
            current_ax.hist(df[column],bins = 'auto', color = '#0070C0', linewidth=1.2)
            current_ax.set_xlabel(f"{column}", fontsize=tick_fontsize)
            current_ax.set_ylabel("Occurences", fontsize=label_fontsize)
            current_ax.set_title(f"Distribution of {column}",fontsize=20,pad=12, )

        if(type == "box"):
            current_ax.boxplot(df[column])
            current_ax.set_xlabel(f"{column}")

        if(type == "bar"):
            value_counts = df[column].value_counts().sort_index()
            current_ax.bar(value_counts.index.astype(str), value_counts.values)
            current_ax.set_xlabel(f"{column}")
            current_ax.set_ylabel("Count")

        current_ax.set_title(f"Distribution of {column}")
        current_ax.grid(True, linestyle='--', alpha=0.6)

        for spine in ['top','right']:
            current_ax.spines[spine].set_visible(False)
        for spine in ['bottom','left']:
            current_ax.spines[spine].set_linewidth(1.2)
            

    for i in range(len(columns), len(axes_flat)):
        axes_flat[i].set_visible(False)
        
    if(save_path):
        plt.savefig(f"./figures/{save_path}.png", dpi = 300)

    plt.show()
    
def plot_bivariate_distribution(df: pd.DataFrame, columns:list, target : str, type = "hist",nrows = 2, ncols = 2, save_path = ""):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    axes_flat = ax.flatten()

    for i, column in enumerate(columns):
        current_ax = axes_flat[i]

        if(type == "box"):
            sns.boxplot(x=target, y=column, data=df, ax=current_ax)
            current_ax.set_title(f"{column} by {target}")
            current_ax.set_xlabel(target)
            current_ax.set_ylabel(column)
        elif type == "bar":
            grouped = (
                df.groupby([column, target])
                .size()
                .reset_index(name="count")
            )
            sns.barplot(data=grouped, x=column, y="count", hue=target, ax=current_ax)
            current_ax.set_title(f"{column} count by {target}")
            current_ax.set_xlabel(column)
            current_ax.set_ylabel("Count")
            current_ax.legend(title=target)


    
    for i in range(len(columns), len(axes_flat)):
        axes_flat[i].set_visible(False)
        
    if(save_path):
        plt.savefig(f"./figures/{save_path}.png", dpi = 300)

    plt.show()

def plot_actual_vs_predicted_scatter(
    y_true, y_pred, save_path=None,
    title="Actual vs Predicted", figsize=(6,6),
    title_fontsize=16, label_fontsize=14, tick_fontsize=12
):
    plt.rcParams.update({
        'figure.facecolor':'none',
        'axes.facecolor':'white',
        'savefig.facecolor':'none',
        'font.family':'sans-serif'
    })
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, color='#229B5A', edgecolor='white', s=50, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, '--', color='#333333', linewidth=1)
    ax.set_title(title, fontsize=title_fontsize, pad=12)
    ax.set_xlabel("Actual", fontsize=label_fontsize)
    ax.set_ylabel("Predicted", fontsize=label_fontsize)
    ax.grid(color='#DDDDDD', linestyle='-', linewidth=0.8)
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_linewidth(1.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_actual_vs_predicted_timeseries(
    y_true, y_pred, index=None, save_path=None,
    title="Actual vs Predicted", xlabel="Time", ylabel="Value",
    figsize=(12,4), title_fontsize=16,
    label_fontsize=14, tick_fontsize=12, legend_fontsize=12
):
    plt.rcParams.update({
        'figure.facecolor':'none',
        'axes.facecolor':'white',
        'savefig.facecolor':'none',
        'font.family':'sans-serif'
    })
    fig, ax = plt.subplots(figsize=figsize)
    x = index if index is not None else np.arange(len(y_true))
    ax.plot(x, y_true,   label="Actual",   color='#0070C0', linewidth=2)
    ax.plot(x, y_pred,   label="Predicted",color='#ED7D31', linewidth=2, alpha=0.8)
    ax.set_title(title,   fontsize=title_fontsize, pad=12)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.legend(frameon=False, fontsize=legend_fontsize)
    ax.grid(axis='y', color='#DDDDDD', linestyle='-', linewidth=0.8)
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_linewidth(1.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300)
    plt.close(fig)


    
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.rcParams.update({
        'figure.facecolor':'none',
        'axes.facecolor':'white',
        'font.family':'sans-serif'
    })
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual',    fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return fig, ax


def plot_roc_curve(y_true, y_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.rcParams.update({
        'figure.facecolor':'none',
        'axes.facecolor':'white'
    })
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2, color='#0070C0', label=f'AUC = {roc_auc:.2f}')
    ax.plot([0,1],[0,1], '--', color='gray')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, pad=10)
    ax.legend(loc='lower right', frameon=False, fontsize=12)
    ax.grid(color='#DDDDDD', linestyle='-', linewidth=0.8)
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return fig, ax

def plot_precision_recall_curve(y_true, y_proba, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.rcParams.update({'figure.facecolor':'none','axes.facecolor':'white'})
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(recall, precision, color='#ED7D31', lw=2, label=f'AP = {ap:.2f}')
    ax.set_xlabel('Recall',    fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision–Recall Curve', fontsize=14, pad=10)
    ax.legend(loc='lower left', frameon=False, fontsize=12)
    ax.grid(color='#DDDDDD', linestyle='-', linewidth=0.8)
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return save_path


def plot_model_comparison(
    df,
    top_n=5,
    metrics=None,
    chart_type='bar',  
    figure_size=(8,4),
    title_fontsize=16,
    label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    save_path=None
):
    """
    Plot a PPT-friendly comparison of model performance.

    Args:
        df           : pandas DataFrame with columns ['Model', ...metrics]
        top_n        : how many top models (by F1 Score) to include
        metrics      : list of metric column names (default ['Accuracy','F1 Score','Precision','Recall'])
        chart_type   : 'bar' for horizontal bar chart; 'radar' for radar/spider plot
        figure_size  : tuple, e.g. (8,4)
        title_fontsize, label_fontsize, tick_fontsize, legend_fontsize : int
        save_path    : if given, path to save (png/svg); transparency enabled

    Returns:
        fig, ax : the matplotlib figure and axes
    """
    if metrics is None:
        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

    plt.rcParams.update({
        'figure.facecolor': 'none',
        'axes.facecolor':   'white',
        'savefig.facecolor':'none',
        'font.family':      'sans-serif',
    })

    df_top = df.sort_values('Recall', ascending=False).head(top_n).reset_index(drop=True)

    if chart_type == 'bar':
        fig, ax = plt.subplots(figsize=figure_size)
        colors = ['#0070C0', '#ED7D31', '#229B5A', '#7030A0']  # your PPT palette
        df_top.plot.barh(
            x='Model',
            y=metrics,
            ax=ax,
            color=colors[:len(metrics)],
            edgecolor='white'
        )
        ax.invert_yaxis()  # highest at top
        ax.legend(loc='lower right', frameon=False, fontsize=legend_fontsize)
        ax.set_xlabel('Score', fontsize=label_fontsize)
        ax.set_title('Model Comparison', fontsize=title_fontsize, pad=12)
        ax.grid(axis='x', color='#DDDDDD', linestyle='-', linewidth=0.8)
        ax.set_xlim(0,1)
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom','left']:
            ax.spines[spine].set_linewidth(1.2)

    elif chart_type == 'radar':
        labels = metrics
        N = len(labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=figure_size,
            subplot_kw=dict(polar=True)
        )

        for _, row in df_top.iterrows():
            values = row[metrics].tolist()
            values += values[:1]
            ax.plot(angles, values, label=row['Model'], linewidth=2)
            ax.fill(angles, values, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=tick_fontsize)
        ax.set_ylim(0,1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1),
                  fontsize=legend_fontsize, frameon=False)
        ax.set_title('Model Performance Radar', fontsize=title_fontsize, pad=20)
        ax.grid(color='#DDDDDD', linestyle='-', linewidth=0.8)

    else:
        raise ValueError("chart_type must be 'bar' or 'radar'")

    plt.tight_layout()

    if save_path:
        ext = save_path.split('.')[-1].lower()
        fig.savefig(
            save_path,
            format=ext,
            bbox_inches='tight',
            transparent=True,
            dpi=300 if ext in ['png','jpg'] else None
        )

    plt.show()
    return fig, ax


def plot_ts_model_comparison(
    df,
    metrics=None,
    sort_by='MSE',
    ascending=True,
    chart_type='bar',        # 'bar' or 'radar'
    figure_size=(8,4),
    title_fontsize=16,
    label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    save_path=None
):
    """
    Plot PPT-friendly comparison of time-series model metrics.

    Args:
        df         : pd.DataFrame with at least ['Model','MSE','MAE','R^2']
        metrics    : list of metrics to plot, default ['MSE','MAE','R^2']
        sort_by    : which metric to sort models by (best on top)
        ascending  : True=lower is better (for MSE/MAE), False=higher is better (for R^2)
        chart_type : 'bar' (default) or 'radar'
        figure_size: tuple
        *_fontsize : ints for styling
        save_path  : if provided, path (png/svg) to save with transparency
    """
    # defaults
    if metrics is None:
        metrics = ['MSE','MAE','R^2']

    # PPT-style rcParams
    plt.rcParams.update({
        'figure.facecolor': 'none',
        'axes.facecolor':   'white',
        'savefig.facecolor':'none',
        'font.family':      'sans-serif',
    })

    # sort & reset
    df_plot = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    if chart_type == 'bar':
        fig, ax = plt.subplots(figsize=figure_size)
        colors = ['#0070C0','#ED7D31','#229B5A']  # one per metric
        df_plot.plot.barh(
            x='Model',
            y=metrics,
            ax=ax,
            color=colors[:len(metrics)],
            edgecolor='white'
        )
        ax.invert_yaxis()
        ax.set_xlabel('Score', fontsize=label_fontsize)
        ax.set_title('Time-Series Model Comparison', fontsize=title_fontsize, pad=12)
        ax.legend(loc='lower right', frameon=False, fontsize=legend_fontsize)
        ax.grid(axis='x', color='#DDDDDD', linestyle='-', linewidth=0.8)
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom','left']:
            ax.spines[spine].set_linewidth(1.2)

    elif chart_type == 'radar':
        # radar won't look ideal with negative R2, but included for completeness
        labels = metrics
        N = len(labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=figure_size)
        for _, row in df_plot.iterrows():
            vals = [row[m] for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, label=row['Model'], linewidth=2)
            ax.fill(angles, vals, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=tick_fontsize)
        ax.set_title('Time-Series Model Radar', fontsize=title_fontsize, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1),
                  frameon=False, fontsize=legend_fontsize)
        ax.grid(color='#DDDDDD', linestyle='-', linewidth=0.8)

    else:
        raise ValueError("chart_type must be 'bar' or 'radar'")

    plt.tight_layout()

    if save_path:
        ext = save_path.split('.')[-1].lower()
        fig.savefig(
            save_path,
            format=ext,
            bbox_inches='tight',
            transparent=True,
            dpi=300 if ext in ['png','jpg'] else None
        )

    plt.show()
    return fig, ax
