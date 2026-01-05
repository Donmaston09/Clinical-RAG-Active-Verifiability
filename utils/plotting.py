import matplotlib.pyplot as plt
import numpy as np

def plot_crts_radar(crts):
    labels = ["Source Fidelity", "Conflict Reporting", "Auditability", "Guideline Alignment"]
    values = [
        crts["source_fidelity"],
        crts["conflict_reporting_rate"],
        crts["auditability_score"],
        crts["guideline_alignment"]
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values += values[:1]
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 1)

    return fig
