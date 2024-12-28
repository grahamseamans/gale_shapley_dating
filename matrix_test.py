import numpy as np
import sys


def main():
    np.set_printoptions(precision=3, suppress=True)

    # Generate three random square matrices A, B, C
    n = 10
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    C = np.random.randn(n, n)

    # Normalize matrices to have unit norm
    A /= np.linalg.norm(A, "fro")
    B /= np.linalg.norm(B, "fro")
    C /= np.linalg.norm(C, "fro")

    print("Normalized Matrix A:\n", A, "\n")
    print("Normalized Matrix B:\n", B, "\n")
    print("Normalized Matrix C:\n", C, "\n")

    # Matrix products for permutations
    combinations = {
        "ABA": A @ B @ A,
        "BAB": B @ A @ B,
        "ACA": A @ C @ A,
        "CAC": C @ A @ C,
        "BCB": B @ C @ B,
        "CBC": C @ B @ C,
    }

    # Initialize lists to store traces and determinants
    trace_list = []
    det_list = []

    # Output traces and determinants for each combination
    for key, matrix in combinations.items():
        trace_val = np.trace(matrix)
        det_val = np.linalg.det(matrix)

        # Check for overlaps in traces
        if trace_val in trace_list:
            print(f"Overlap found in trace for {key} with value: {trace_val}")
        else:
            trace_list.append(trace_val)
            print(f"Trace({key}):", trace_val)

        # Check for overlaps in determinants
        if det_val in det_list:
            print(f"Overlap found in determinant for {key} with value: {det_val}")
        else:
            det_list.append(det_val)
            print(f"Det({key}):", det_val)

        print("\n")

    # Additional matrix operations to explore more properties
    # Commutators for non-commutativity check
    print("Commutator [A,B] = AB - BA:\n", A @ B - B @ A, "\n")
    print("Commutator [A,C] = AC - CA:\n", A @ C - C @ A, "\n")
    print("Commutator [B,C] = BC - CB:\n", B @ C - C @ B, "\n")

    # Eigenvalues for a selected combination
    eigvals_BAB, eigvecs_BAB = np.linalg.eig(combinations["BAB"])
    print("Eigenvalues of BAB:\n", eigvals_BAB, "\n")


if __name__ == "__main__":
    main()
