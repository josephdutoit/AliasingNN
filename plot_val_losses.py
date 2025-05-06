import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1.train import summary_iterator

def extract_tensorboard_data(log_dir):
    """
    Extracts the 'val_loss' data from TensorBoard logs in a directory using tensorflow.compat.v1.summary.summary_iterator.

    Args:
        log_dir (str): The directory containing the TensorBoard logs.

    Returns:
        dict: A dictionary where keys are run names and values are lists of 'val_loss' values for each epoch.
    """
    run_data = {}
    for run_name in sorted(os.listdir(log_dir)):
        run_path = os.path.join(log_dir, run_name, "lightning_logs", "version_0")
        print(f"Processing run: {run_path}")
        if not os.path.isdir(run_path):
            continue

        try:
            loss_values = []
            event_files = tf.io.gfile.glob(os.path.join(run_path, "events.out.tfevents.*"))
            print(f"Event files found in {run_path}:", event_files)
            if not event_files:
                print(f"No event files found in {run_path}")
                continue  # Skip to the next run if no event files are found

            for event_file in event_files:
                try:
                    for event in summary_iterator(event_file):  # Iterate through events files
                        for value in event.summary.value:
                            if value.tag == 'val_loss':
                                loss_values.append(value.simple_value)
                except Exception as e:
                    print(f"Error reading event file {event_file}: {e}")

            run_data[run_name] = loss_values

        except Exception as e:
            print(f"Error processing run {run_name}: {e}")
    return run_data

def plot_val_loss_per_epoch(run_data, output_path="val_loss_per_epoch.png"):
    """
    Plots the validation loss for each epoch for each run.

    Args:
        run_data (dict): A dictionary where keys are run names and values are lists of 'val_loss' values.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))
    for run_name, loss_values in run_data.items():
        plt.plot(loss_values, label=run_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss per Epoch for Each Run")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_best_val_loss(run_data, output_path="best_val_loss.png"):
    """
    Plots the best validation loss for each run as a line plot.

    Args:
        run_data (dict): A dictionary where keys are run names and values are lists of 'val_loss' values.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))
    best_losses = {run_name: min(loss_values) for run_name, loss_values in run_data.items()}
    
    run_names = list(best_losses.keys())
    losses = list(best_losses.values())
    
    plt.plot(run_names, losses, marker='o')  # Use a line plot with markers

    plt.xlabel("Run")
    plt.ylabel("Best Validation Loss")
    plt.title("Best Validation Loss for Each Run")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    log_directory = "logs"  # Replace with your actual log directory
    run_data = extract_tensorboard_data(log_directory)

    if run_data:
        plot_val_loss_per_epoch(run_data)
        plot_best_val_loss(run_data)  # Plot the best validation loss
    else:
        print("No validation loss data found in the logs.")