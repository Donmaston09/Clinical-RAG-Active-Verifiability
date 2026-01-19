
"""
Plotting utilities for CRTS
- Radar chart expects values in [0,1] for: SF, CRR, AR*, GA
"""
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["Source Fidelity", "Conflict Reporting", "Audit Responsiveness", "Guideline Alignment"]


def plot_crts_radar(crts_dict):
    """crts_dict can be either the raw dict from compute_crts or a mapping
    with keys matching LABELS. We normalise accordingly.
    """
    # Flexible input handling
    if all(k in crts_dict for k in ("sf","crr","ar","ga")):
        values = [crts_dict['sf'], crts_dict['crr'], crts_dict['ar'], crts_dict['ga']]
    else:
        # assume a mapping from pretty labels → values
        values = [crts_dict.get(LABELS[0],0), crts_dict.get(LABELS[1],0), crts_dict.get(LABELS[2],0), crts_dict.get(LABELS[3],0)]

    values = np.array(values, dtype=float)
    # close the loop
    vals_closed = np.r_[values, values[0]]
    angles = np.linspace(0, 2*np.pi, len(LABELS), endpoint=False)
    ang_closed = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(angles * 180/np.pi, LABELS)
    ax.set_ylim(0, 1)
    ax.plot(ang_closed, vals_closed, color='#FF1493', linewidth=2)
    ax.fill(ang_closed, vals_closed, color='#FF1493', alpha=0.15)
    return fig
