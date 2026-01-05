import matplotlib.pyplot as plt
import numpy as np

def plot_crts_radar(plot_data):
    # These labels must match the keys in the dictionary you pass from app.py
    labels = ["Source Fidelity", "Conflict Reporting", "Audit Responsiveness", "Guideline Alignment"]

    # Use .get() to avoid KeyErrors if a metric is missing
    values = [
        plot_data.get("Source Fidelity", 0),
        plot_data.get("Conflict Reporting", 0),
        plot_data.get("Audit Responsiveness", 0),
        plot_data.get("Guideline Alignment", 0)
    ]

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we must "complete the loop"
    # by appending the start value to the end.
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color='grey', size=10)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')

    # Fill area
    ax.fill(angles, values, color='#1f77b4', alpha=0.3)

    return fig
