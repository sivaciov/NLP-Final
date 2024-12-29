import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_aggregated_error_heatmap(jsonl_file, group_size=10):
    """
    Reads a .jsonl file with prediction data and generates an aggregated heatmap
    for groups of examples, visualizing the average prediction probabilities.

    :param jsonl_file: Path to the .jsonl file containing the data.
    :param group_size: Number of examples to aggregate per row.
    """
    # Load data from the JSONL file
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Initialize lists to store probabilities
    probabilities = []

    # Process each example
    for example in data:
        predicted_scores = example.get('predicted_scores', [])

        # Skip examples without predicted scores
        if not predicted_scores:
            continue

        # Convert predicted scores to probabilities (softmax)
        scores_exp = np.exp(predicted_scores)
        probabilities.append(scores_exp / np.sum(scores_exp))

    # Convert probabilities to numpy array
    probabilities = np.array(probabilities)

    # Aggregate statistics by grouping
    num_groups = int(np.ceil(len(probabilities) / group_size))
    aggregated_probabilities = [
        probabilities[i * group_size: (i + 1) * group_size].mean(axis=0)
        for i in range(num_groups)
    ]

    # Prepare metadata for y-axis
    y_labels = [f"G{i + 1}" for i in range(num_groups)]  # Shortened labels

    # Plot aggregated heatmap
    plt.figure(figsize=(10, len(y_labels) * 0.5))
    sns.heatmap(
        aggregated_probabilities,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=["Entailment", "Neutral", "Contradiction"],
        yticklabels=y_labels
    )
    plt.title("Aggregated Error Heatmap (Grouped by 10 Examples)")
    plt.xlabel("Classes")
    plt.ylabel("Groups")
    plt.tight_layout()
    plt.savefig("aggregated_error_heatmap.png")
    plt.show()


# Specify the path to your .jsonl file
jsonl_file_path = "./eval_output/eval_predictions.jsonl"

# Generate the aggregated heatmap
plot_aggregated_error_heatmap(jsonl_file_path, group_size=10)
