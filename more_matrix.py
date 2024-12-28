import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_sequence(ax, transformations, title, color_cycle):
    """
    Plots the evolution of the unit square through each step
    in the given sequence of transformations.
    """
    # Define corners of the unit square (column form):
    # (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
    square_corners = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])

    # Start with identity and accumulate transforms
    partial_matrices = [np.eye(2)]
    for T in transformations:
        partial_matrices.append(T @ partial_matrices[-1])

    # Plot each step in a different color from the cycle
    for step_index, mat in enumerate(partial_matrices):
        transformed = mat @ square_corners
        ax.plot(
            transformed[0, :],
            transformed[1, :],
            color=color_cycle[step_index % len(color_cycle)],
            linewidth=2,
        )

    # Optionally, add minimal x- and y-axis arrows
    ax.quiver(0, 0, 1, 0, color="black", angles="xy", scale_units="xy", scale=1)
    ax.quiver(0, 0, 0, 1, color="black", angles="xy", scale_units="xy", scale=1)

    # Style the subplot
    ax.set_aspect("equal", "box")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(title, fontsize=10)
    ax.axis("off")  # Hide axis ticks and lines


N = 10
color_cycle = ["blue", "red", "green", "orange", "purple", "cyan"]

with PdfPages("random_transformations.pdf") as pdf:
    for i in range(N):
        # Create one figure per row
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Random A and B for each row
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)

        # Left subplot: A -> B -> A
        plot_sequence(
            axes[0], [A, B, A], title=f"Row {i+1}: A → B → A", color_cycle=color_cycle
        )

        # Right subplot: B -> A -> B
        plot_sequence(
            axes[1], [B, A, B], title=f"Row {i+1}: B → A → B", color_cycle=color_cycle
        )

        fig.suptitle(f"Random Pair {i+1}", fontsize=12)
        plt.tight_layout()

        # Save this figure (row) as one page in the PDF
        pdf.savefig(fig)
        plt.close(fig)
