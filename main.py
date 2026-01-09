import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt   
import matplotlib.patches as mpatches 
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


import Graph




def task_1_2():
    alphas = [1, 5, 10]
    gg = Graph.GraphGenerator()
    adj_matrices = []

    # compute the matrices and plot them
    for alpha in alphas:
        A = gg.Kaiser_and_Hilgetag(alpha=alpha)
        G = nx.from_numpy_array(A)

        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(A, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax1.set_title(f"Adjacency (alpha={alpha})")
        off = mpatches.Patch(color=plt.cm.Blues(0.15), label="0")
        on  = mpatches.Patch(color=plt.cm.Blues(0.85), label="1")
        ax1.legend(handles=[off, on], loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=True)

        ax2 = plt.subplot(1, 2, 2)
        nx.draw(G, pos=nx.spring_layout(G, seed=42), node_size=30, width=0.5, with_labels=False, ax=ax2)
        ax2.set_title("Graph")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

def task_1_3():
    beta, N, logger, iterations = 3, 100, 0, 10
    alphas = np.logspace(-1, 2, 10) # we try to compute these 1k vals, see how long it takes. Edit: a minute or so.
    gg = Graph.GraphGenerator()
    clustering_coeffs, ASPs = [], []
    for alpha in alphas:
        total_clustering, total_ASPs = 0, 0
        for i in range(iterations):
            A = gg.Kaiser_and_Hilgetag(alpha=alpha, beta=1)
            G = nx.from_numpy_array(A)
            total_clustering += nx.average_clustering(G)
            total_ASPs += nx.average_shortest_path_length(G)
    
        clustering_coeffs.append(total_clustering / iterations)
        ASPs.append(total_ASPs / iterations)

        
    df = pd.DataFrame({
        "alpha": alphas,
        "clustering_coeff": clustering_coeffs,
        "ASP": ASPs
    })


    df.to_csv("task_1_3_results.csv", index=False)
    alphas = df["alpha"].to_numpy()
    clustering_coeffs = df["clustering_coeff"].to_numpy()
    ASPs = df["ASP"].to_numpy()


    fig, ax1 = plt.subplots(figsize=(6, 4))

    # x: alpha (log scale)
    ax1.set_xscale("log")

    # left y: clustering coefficient C (blue)
    ax1.plot(alphas, clustering_coeffs, color="tab:blue", marker="^", markersize=3, linewidth=1)
    ax1.set_ylabel("C", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # right y: average shortest path (red)
    ax2 = ax1.twinx()
    ax2.plot(alphas, ASPs, color="tab:red", marker="D", markersize=3, linewidth=1)
    ax2.set_ylabel("ASP", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_xlabel("alpha")
    ax1.set_title("Parameter scan (beta=1, N=100)")
    fig.tight_layout()
    plt.show()


def task_2_2():
    alphas = [5, 10, 20]
    N = 100
    gg = Graph.GraphGenerator()

   

    for alpha in alphas:
        A, W = gg.Vertes(alpha=alpha, N=N)
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)

        
        off = mpatches.Patch(color=plt.cm.Blues(0.15), label="0")
        on  = mpatches.Patch(color=plt.cm.Blues(0.85), label="1")
        plt.figure(figsize=(10, 4))

        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(A, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax1.set_title(f"Adjacency (alpha={alpha})")
        ax1.legend(handles=[off, on], loc="upper left",
                   bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=True)

        ax2 = plt.subplot(1, 2, 2)
        nx.draw(G, pos=nx.spring_layout(G, seed=42),
                node_size=30, width=0.5, with_labels=False, ax=ax2,
                arrows=True) 
        ax2.set_title("Graph")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()
    

def task_2_3():
    """A comment: I first thought that both the algorithms take place in the [0,1] interval, i.e. confused 2D and 1D.
    And found out at the day of the deadline. LOL.
    In 1D the graphs produced by the vertes algorithm became stronlgy disconnected, so thats why I checked it here."""
    alphas = np.logspace(-1, 1.5, 10) # we try to compute these values, this time only 10 of them, which suffice for the plot
    N, logger, iterations = 50, 0, 3
    gg = Graph.GraphGenerator()
    clustering_coeffs, ASPs, ASPs_largest_component = [], [], []
    for alpha in alphas:
        total_clustering, total_ASPs, total_ASPs_largest_component = 0, 0, 0
        for i in range(iterations):
            A, W = gg.Vertes(alpha=alpha, N=100)
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            total_clustering += nx.average_clustering(G)

            if nx.is_strongly_connected(G):
                l = nx.average_shortest_path_length(G)
                total_ASPs += l
                total_ASPs_largest_component += l
            else:
                total_ASPs = float('inf')
                # take the ASPs number from the largest component as a substitute
                largest_component_nodes = max(nx.strongly_connected_components(G), key=len)
                largest_component = G.subgraph(largest_component_nodes).copy()
                total_ASPs_largest_component += nx.average_shortest_path_length(largest_component)
                break

        clustering_coeffs.append(total_clustering / iterations)
        ASPs.append(total_ASPs / iterations)
        ASPs_largest_component.append(total_ASPs_largest_component / iterations)

        logger += 1
        print(f"Processed {logger} / {len(alphas)} alphas.")
        
    df = pd.DataFrame({
        "alpha": alphas,
        "clustering_coeff": clustering_coeffs,
        "ASP": ASPs,
        "ASP_largest_component": ASPs_largest_component
    })
    df.to_csv("task_2_3_results.csv", index=False)



def plotting_2_3():
    df = pd.read_csv("task_2_3_results.csv")
    alphas = df["alpha"].to_numpy()
    C = df["clustering_coeff"].to_numpy()
    L = df["ASP"].to_numpy()

    finite = np.isfinite(L)
    missing_alphas = alphas[~finite]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xscale("log")

    # plot clustering only where ASP is finite
    ax1.plot(alphas[finite], C[finite], color="tab:blue", marker="^", markersize=3, linewidth=1)
    ax1.set_ylabel("C", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # plot ASP only where finite
    ax2 = ax1.twinx()
    ax2.plot(alphas[finite], L[finite], color="tab:red", marker="D", markersize=3, linewidth=1)
    ax2.set_ylabel("ASP", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # rug plot for missing ASP values
    if missing_alphas.size > 0:
        ax1.plot(
            missing_alphas,
            np.zeros_like(missing_alphas),
            "|",
            transform=ax1.get_xaxis_transform(),
            color="tab:red",
            markersize=8,
            markeredgewidth=1.2,
            clip_on=False,
        )

    ax1.set_xlabel("alpha")
    ax1.set_title("Parameter scan (N=100, density=0.1)")
    fig.tight_layout()
    plt.show()

def task_2_4():
    alpha, N = 15, 100
    gg = Graph.GraphGenerator()

    A, W = gg.Vertes(alpha=alpha, N=N)

    out_deg = A.sum(axis=1)
    in_deg  = A.sum(axis=0)

    weights = W.ravel()
    print(weights)
    plt.figure(figsize=(10, 4))

    # Degree distribution: stacked in/out histograms
    plt.subplot(1, 2, 1)
    bins_deg = np.arange(min(in_deg.min(), out_deg.min()), max(in_deg.max(), out_deg.max()) + 2) - 0.5
    plt.hist([in_deg, out_deg], bins=bins_deg, stacked=True, edgecolor="black", label=["In-degree", "Out-degree"])
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title(f"Degree distribution (alpha={alpha}, N={N})")
    plt.legend()
    # Weight distribution (log y-scale)
    plt.subplot(1, 2, 2)
    bins_w = np.arange(1, weights.max() + 2) - 0.5
    plt.hist(weights, bins=bins_w, edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Edge weight")
    plt.ylabel("Count (log scale)")
    plt.title("Weight distribution")

    plt.tight_layout()
    plt.show()


def task_3_1():
    A = np.load("mouse_V1_adjacency_matrix.npy")

    # Treat any positive entry as an edge (keeps compatibility if A is weighted)
    A_bin = (A > 0).astype(int)

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    N = A.shape[0]
    E = int(np.count_nonzero(A_bin))
    density = E / (N * (N - 1))
    total_weight = float(A.sum())

    print(f"Mouse network: N={N}, E={E}, density={density:.3f}, total_weight={total_weight:.1f}")

    # Degrees (edge-count degrees, not strength)
    out_deg = A_bin.sum(axis=1)
    in_deg  = A_bin.sum(axis=0)

    # Weights (only existing edges)
    weights = A[A > 0].ravel()

    plt.figure(figsize=(12, 8))

    # --- (1) adjacency matrix ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(A_bin, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax1.set_title("Adjacency matrix (binary view)")
    off = mpatches.Patch(color=plt.cm.Blues(0.15), label="0")
    on  = mpatches.Patch(color=plt.cm.Blues(0.85), label="1")
    ax1.legend(handles=[off, on], loc="upper left", bbox_to_anchor=(1.02, 1),
               borderaxespad=0., frameon=True)

    # --- (2) network drawing ---
    ax2 = plt.subplot(2, 2, 2)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos=pos, node_size=25, width=0.4, with_labels=False, ax=ax2, arrows=True)
    ax2.set_title("Mouse Visual Cortex Network")
    ax2.axis("off")

    # --- (3) degree distribution (stacked in/out) ---
    ax3 = plt.subplot(2, 2, 3)
    max_deg = int(max(in_deg.max(), out_deg.max()))
    bins_deg = np.arange(0, max_deg + 2) - 0.5
    ax3.hist([in_deg, out_deg], bins=bins_deg, stacked=True,
             edgecolor="black", label=["In-degree", "Out-degree"])
    ax3.set_xlabel("Degree")
    ax3.set_ylabel("Count")
    ax3.set_title("Degree distribution (in/out, stacked)")
    ax3.legend()

    # --- (4) weight distribution (log y) ---
    ax4 = plt.subplot(2, 2, 4)
    if weights.size == 0:
        ax4.text(0.5, 0.5, "No edges / no weights", ha="center", va="center")
        ax4.set_axis_off()
    else:
        w_min, w_max = float(weights.min()), float(weights.max())

        # If weights are integer-like, use integer bins; otherwise fall back to automatic bins
        if np.allclose(weights, np.round(weights)):
            w_min_i, w_max_i = int(np.round(w_min)), int(np.round(w_max))
            bins_w = np.arange(w_min_i, w_max_i + 2) - 0.5
            ax4.hist(weights, bins=bins_w, edgecolor="black")
        else:
            ax4.hist(weights, bins=30, edgecolor="black")

        ax4.set_yscale("log")
        ax4.set_xlabel("Edge weight")
        ax4.set_ylabel("Count (log scale)")
        ax4.set_title("Weight distribution")

    plt.tight_layout()
    plt.show()


def task_3_2():
    A = np.load("mouse_V1_adjacency_matrix.npy")

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    largest_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_nodes).copy()

    N = G.number_of_nodes()
    E = G.number_of_edges()
    density = nx.density(G) 

    C = nx.average_clustering(G)
    ASPs = nx.average_shortest_path_length(G)

    print("Mouse Visual Cortex (largest SCC)")
    print(f"N = {N}")
    print(f"E = {E}")
    print(f"density = {density:.4f}")
    print(f"C (avg clustering) = {C:.4f}")
    print(f"ASP (avg shortest path) = {ASPs:.4f}")



def task_3_2_KH():
    A = np.load("mouse_V1_adjacency_matrix.npy")

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    largest_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_nodes).copy()

    

    N_target = G.number_of_nodes()

    C_mouse = nx.average_clustering(G)
    L_mouse = nx.average_shortest_path_length(G) 
    rho_mouse = nx.density(G)
    E_mouse = G.number_of_edges()


    print(f"N={N_target}, density={rho_mouse:.4f}, C={C_mouse:.4f}, ASP={L_mouse:.4f}, E={E_mouse}")

    wC = 1.0
    wRho = 5.0
    wL = 1.0 / 5.0

    betas = np.logspace(-2, 0, 5)
    alphas = np.logspace(0, 2, 5)

    iterations = 3
    gg = Graph.GraphGenerator()

    results = []

    for alpha in alphas:
        for beta in betas:
            losses = []
            Cs, Ls, rhos = [], [], []

            for _ in range(iterations):
                A_kh = gg.Kaiser_and_Hilgetag(alpha=alpha, beta=beta, N=N_target)
                KH = nx.from_numpy_array(A_kh)  

                C_kh = nx.average_clustering(KH)
                rho_kh = nx.density(KH)
                L_kh = nx.average_shortest_path_length(KH)

                loss = (
                    wC * abs(C_kh - C_mouse)
                    + wL * abs(L_kh - L_mouse)
                    + wRho * abs(rho_kh - rho_mouse)
                )
 

                Cs.append(C_kh)
                Ls.append(L_kh)
                rhos.append(rho_kh)
                losses.append(loss)

            # Average the LOSS over 5 measurements (as you described)
            loss_mean = float(np.mean(losses))

            results.append({
                "alpha": float(alpha),
                "beta": float(beta),
                "loss": loss_mean,
                "C_mean": float(np.mean(Cs)),
                "ASP_mean": float(np.mean([x for x in Ls if np.isfinite(x)]) if any(np.isfinite(Ls)) else np.inf),
                "density_mean": float(np.mean(rhos)),
                "loss_std": float(np.std(losses)),
                "C_std": float(np.std(Cs)),
                "density_std": float(np.std(rhos)),
            })

    df = pd.DataFrame(results).sort_values("loss", ascending=True).reset_index(drop=True)

    print("\nTop 5 parameter settings by loss:")
    for i, row in df.head(5).iterrows():
        print(
            f"{i+1}) alpha={row['alpha']:.4g}, beta={row['beta']:.4g}, "
            f"loss={row['loss']:.6g} | C={row['C_mean']:.4f}, ASP={row['ASP_mean']:.4f}, rho={row['density_mean']:.4f}"
        )

    df.to_csv("task_3_2_KH_scan.csv", index=False)

    # Now plot it as a 3D grid
    Agrid, Bgrid = np.meshgrid(alphas, betas, indexing="ij")

    Z = df.pivot(index="alpha", columns="beta", values="loss").to_numpy()
    Zmask = np.isfinite(Z)

    norm = Normalize(vmin=np.nanmin(Z[Zmask]), vmax=np.nanmax(Z[Zmask]))
    facecolors = cm.viridis(norm(Z))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        np.log10(Agrid), np.log10(Bgrid), Z,
        facecolors=facecolors, rstride=1, cstride=1,
        linewidth=0, antialiased=True, shade=False
    )

    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array(Z[Zmask])
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1, label="Loss")

    ax.set_xlabel(r"$\log_{10}(\alpha)$")
    ax.set_ylabel(r"$\log_{10}(\beta)$")
    ax.set_zlabel("Loss")
    ax.set_title("KH scan: Loss over (alpha, beta)")

    plt.tight_layout()
    plt.show()



def task_3_2_Vertes():

    A = np.load("mouse_V1_adjacency_matrix.npy")

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    largest_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_nodes).copy()

    N_target = G.number_of_nodes()
    rho_mouse = nx.density(G)

    C_mouse = nx.average_clustering(G)
    L_mouse = nx.average_shortest_path_length(G)

    print(f"Mouse SCC: N={N_target}, density={rho_mouse:.4f}, C={C_mouse:.4f}, ASP={L_mouse:.4f}")


    wC = 1.0
    wL = 1.0 / 5.0

    alphas = np.logspace(-1, 1.5, 50)
    iterations = 3

    gg = Graph.GraphGenerator()
    results = []

    for alpha in alphas:
        Cs, Ls, losses = [], [], []
        disconnected = False

        for i in range(iterations):
            A_v, W_v = gg.Vertes(alpha=alpha, N=N_target, density=rho_mouse)
            V = nx.from_numpy_array(A_v, create_using=nx.DiGraph)

            if not nx.is_strongly_connected(V):
                disconnected = True
                break

            C_v = nx.average_clustering(V)
            L_v = nx.average_shortest_path_length(V)

            loss = wC * abs(C_v - C_mouse) + wL * abs(L_v - L_mouse)

            Cs.append(C_v)
            Ls.append(L_v)
            losses.append(loss)

        if disconnected:
            results.append({
                "alpha": float(alpha),
                "loss": np.nan,
                "C_mean": np.nan,
                "ASP_mean": np.nan,
                "density_used": float(rho_mouse),
                "disconnected": True
            })
        else:
            results.append({
                "alpha": float(alpha),
                "loss": float(np.mean(losses)),
                "C_mean": float(np.mean(Cs)),
                "ASP_mean": float(np.mean(Ls)),
                "density_used": float(rho_mouse),
                "disconnected": False
            })

    df = pd.DataFrame(results)

    df_finite = df[np.isfinite(df["loss"])].sort_values("loss", ascending=True).reset_index(drop=True)
    print("\nTop 5 alpha settings by loss (connected only):")
    for i, row in df_finite.head(5).iterrows():
        print(f"{i+1}) alpha={row['alpha']:.4g}, loss={row['loss']:.6g} | C={row['C_mean']:.4f}, ASP={row['ASP_mean']:.4f}")

    df_sorted = df.sort_values("loss", ascending=True, na_position="last").reset_index(drop=True)
    df_sorted.to_csv("task_3_2_Vertes_scan.csv", index=False)
    print("\nSaved full results to: task_3_2_Vertes_scan.csv")

    plt.figure(figsize=(6, 4))
    plt.xscale("log")
    plt.plot(df["alpha"], df["loss"], marker="o", markersize=3, linewidth=1)
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.title("Vertes scan: loss over alpha")
    plt.tight_layout()
    plt.show()

    return df_sorted



task_3_2_Vertes()
