import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_plots(df, column):
    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df.index, df[column])

    # Define the locator as before to control tick frequency
    locator = mdates.MonthLocator() # Explicitly locate ticks at the start of each month
    
    # Replace ConciseDateFormatter with the explicit DateFormatter
    formatter = mdates.DateFormatter('%b %Y') # Format: Abbreviated Month and Full Year

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    fig.autofmt_xdate(rotation=45) # rotation for better spacing
    
    plt.show()

def draw_plots_grid(df, columns,  windows=[7, 28],save_path = "",nrows = 2, ncols = 2):

    df.index = pd.to_datetime(df.index)

    # Create a 2x2 grid of subplots. 'ax' is now a 2x2 NumPy array.
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))

    # Flatten the 2D array of axes to a 1D array for easy iteration.
    axes_flat = ax.flatten()

    # Loop through each column and its corresponding subplot axis.
    for i, col_name in enumerate(columns):
        # Select the current axis from the flattened array.
        current_ax = axes_flat[i]
        
        # Plot the data on the current axis.
        current_ax.plot(df.index, df[col_name], label = "original")
        colors = ['orange', 'red', 'green'] 
        for j, window_size in enumerate(windows):
            rolling_mean = df[col_name].rolling(window=window_size).mean()
            current_ax.plot(df.index, rolling_mean, label=f'{window_size}-Day Avg', color=colors[j], linewidth=2 + j)
        
        # Set a title for the individual subplot.
        current_ax.set_title(f"Time Series of {col_name}")

        # Define locator and formatter for THIS specific subplot's x-axis.
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter('%b %Y')

        current_ax.xaxis.set_major_locator(locator)
        current_ax.xaxis.set_major_formatter(formatter)
        current_ax.legend()
        current_ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout(pad=3.0)
    
    for i in range(len(columns), len(axes_flat)):
        axes_flat[i].set_visible(False)

    if(save_path):
        plt.savefig(save_path, dpi = 300)
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