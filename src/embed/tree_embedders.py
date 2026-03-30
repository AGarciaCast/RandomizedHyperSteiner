import biotite.sequence.phylo as phylo
import networkx as nx

import numpy as np
import torch
from src.embed.distances import *
from src.embed.embedders import *
from src.embed.optimizers import *



def neighbor_joining(points, dist_matrix):
    
    tree = phylo.neighbor_joining(dist_matrix)
    G = tree.as_graph().to_undirected()
    
    mapping = dict()
    count = 0
    for n in G.nodes:
        if type(n) is tuple:
            mapping[n] = f"S{count}"
            count+=1
        else:
            mapping[n] = f"T{n}"
    
    G = nx.relabel_nodes(G, mapping)

    return nx.get_edge_attributes(G,'distance'), G


def adjacency_matrix(graph_networkx):

    terminal_nodes = sorted([n for n in graph_networkx.nodes if n.startswith('T')], 
                           key=lambda x: int(x[1:]))
    steiner_nodes = sorted([n for n in graph_networkx.nodes if n.startswith('S')], 
                          key=lambda x: int(x[1:]))
    
    # Create the ordered list: T0, T1, ..., Tlast, S0, S1, ..., Slast
    nodes_ordered = terminal_nodes + steiner_nodes
    
    # Create mapping from node label to matrix index
    node_to_index = {node: i for i, node in enumerate(nodes_ordered)}
    
    # Initialize adjacency matrix
    nbr_nodes = len(nodes_ordered)
    adjacency_matrix = np.zeros((nbr_nodes, nbr_nodes))
    
    # Fill the adjacency matrix
    for edge in graph_networkx.edges:
        u, v = edge
        i = node_to_index[u]
        j = node_to_index[v]
        adjacency_matrix[i][j] = 1
        adjacency_matrix[j][i] = 1  # Undirected graph
    
    return adjacency_matrix, nodes_ordered   



# def train_steiner_embeddings(adjacency_matrix, terminals_poincare, num_epochs=100, lr=0.1, steiners_poincare = None, verbose=True, plot=True):

#     T = len(terminals_poincare)
#     N = adjacency_matrix.shape[0]
#     S = N-T
    
#     model = Embedder(data_size=S, latent_dim=2, distr='hypergaussian', sigma=0.1, steiners_poincare = steiners_poincare)
#     optimizer = PoincareOptim(model, lr=lr)
  
#     total_loss = []

#     for epoch in range(num_epochs):
        
#         model.normalize()
#         steiners = model.embeddings
#         combined_embeddings = torch.cat([terminals_poincare, steiners], dim=0)

#         # dist_matrix = distance_matrix(combined_embeddings, poincare_distance)   
#         # loss = 0.5*((dist_matrix*adjacency_matrix).mean())


#         pairs = torch.nonzero(adjacency_matrix == 1, as_tuple=True) 
#         row_indices, col_indices = pairs 
#         pairs_distances = poincare_distance(
#                     combined_embeddings[row_indices], 
#                     combined_embeddings[col_indices]
#                 )     
#         loss = 0.5*((pairs_distances).mean())


#         total_loss.append(loss.item())


#         # Plot every num_epochs//10 epochs to have 10 plots in total
#         if verbose and ((epoch==num_epochs-1)):
#             print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")
            
#             if  plot:
#                 # Create plot
#                 plt.figure(figsize=(15, 5))
                
#                 # Plot 1: Loss history
#                 plt.subplot(1, 3, 1)
#                 plt.plot(np.arange(len(total_loss)), total_loss)
#                 plt.title(f'Training Loss (Epoch {epoch})')
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Loss')
#                 plt.grid(True)
                
#                 # Plot 2: Current embeddings
#                 plt.subplot(1, 3, 2)
                
#                 terminals_ = terminals_poincare.detach().cpu().numpy()
#                 steiners_ = steiners.detach().cpu().numpy()
#                 combined_embeddings_ = combined_embeddings.detach().cpu().numpy()
#                 plt.scatter(terminals_[:, 0], terminals_[:, 1], 
#                         c='red', label='Terminals', s=50, alpha=0.7)
#                 plt.scatter(steiners_[:, 0], steiners_[:, 1], 
#                         c='blue', label='Steiner', s=30, alpha=0.7)


#                 for i in range(N):
#                     for j in range(N):
#                         if adjacency_matrix[i, j] == 1:
#                             plt.plot([combined_embeddings_[i, 0], combined_embeddings_[j, 0]], [combined_embeddings_[i, 1], combined_embeddings_[j, 1]], 'k--')

#                 # Unit circle boundary
#                 # circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
#                 # plt.gca().add_artist(circle)
#                 # plt.xlim(-1.1, 1.1)
#                 # plt.ylim(-1.1, 1.1)
                
#                 plt.title(f'Embeddings at Epoch {epoch}')
#                 plt.xlabel('X')
#                 plt.ylabel('Y')
#                 plt.legend()
#                 plt.axis('equal')
#                 plt.grid(True)
                
                
#                 plt.tight_layout()
#                 plt.show()
                
#                 # plt.savefig(f'epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
#                 plt.close()  


#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step(np.arange(S))

#     model.normalize()

#     return model.embeddings



def train_steiner_embeddings(adjacency_matrix, terminals_poincare, num_epochs=100, lr=0.1, 
                           steiners_poincare=None, verbose=True, plot=True,
                           # Early stopping parameters
                           early_stopping=False, patience=10, min_delta=1e-6, 
                           restore_best_weights=True):
    """
    Train Steiner embeddings with optional early stopping.
    
    Args:
        adjacency_matrix: Graph adjacency matrix
        terminals_poincare: Terminal node embeddings
        num_epochs: Maximum number of epochs
        lr: Learning rate
        steiners_poincare: Initial Steiner embeddings (optional)
        verbose: Print training progress
        plot: Show training plots
        early_stopping: Enable early stopping
        patience: Number of epochs to wait after loss stops improving
        min_delta: Minimum change in loss to qualify as an improvement
        restore_best_weights: Whether to restore best weights when early stopping
    
    Returns:
        Final embeddings (either from last epoch or best epoch if early stopping)
    """

    T = len(terminals_poincare)
    N = adjacency_matrix.shape[0]
    S = N-T
    
    # If there are no Steiner nodes to train, return empty tensor
    if S == 0:
        return torch.empty(0, 2, dtype=terminals_poincare.dtype, device=terminals_poincare.device)
    
    model = Embedder(data_size=S, latent_dim=2, distr='hypergaussian', 
                    sigma=0.1, steiners_poincare=steiners_poincare)
    optimizer = PoincareOptim(model, lr=lr)
  
    total_loss = []
    
    # Early stopping variables
    if early_stopping:
        best_loss = float('inf')
        patience_counter = 0
        best_embeddings = None
        stopped_epoch = 0

    for epoch in range(num_epochs):
        
        model.normalize()
        steiners = model.embeddings
        combined_embeddings = torch.cat([terminals_poincare, steiners], dim=0)

        # Calculate loss
        pairs = torch.nonzero(adjacency_matrix == 1, as_tuple=True) 
        row_indices, col_indices = pairs 
        pairs_distances = poincare_distance(
                    combined_embeddings[row_indices], 
                    combined_embeddings[col_indices]
                )     
        loss = 0.5*((pairs_distances).mean())

        total_loss.append(loss.item())
        
        # Early stopping logic
        if early_stopping:
            if loss.item() < best_loss - min_delta:
                best_loss = loss.item()
                patience_counter = 0
                if restore_best_weights:
                    best_embeddings = model.embeddings.clone()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                stopped_epoch = epoch
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Best loss: {best_loss:.6f}")

                # Progress reporting
                    print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")
                    
                    if plot:
                        # Create plot
                        plt.figure(figsize=(15, 5))
                        
                        # Plot 1: Loss history
                        plt.subplot(1, 3, 1)
                        plt.plot(np.arange(len(total_loss)), total_loss)
                        if early_stopping:
                            plt.axvline(x=len(total_loss) - patience_counter - 1, 
                                    color='red', linestyle='--', alpha=0.7, label='Best')
                        plt.title(f'Training Loss (Epoch {epoch})')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.grid(True)
                        
                        # Plot 2: Current embeddings
                        plt.subplot(1, 3, 2)
                        
                        terminals_ = terminals_poincare.detach().cpu().numpy()
                        steiners_ = steiners.detach().cpu().numpy()
                        combined_embeddings_ = combined_embeddings.detach().cpu().numpy()
                        plt.scatter(terminals_[:, 0], terminals_[:, 1], 
                                c='red', label='Terminals', s=50, alpha=0.7)
                        plt.scatter(steiners_[:, 0], steiners_[:, 1], 
                                c='blue', label='Steiner', s=30, alpha=0.7)

                        for i in range(N):
                            for j in range(N):
                                if adjacency_matrix[i, j] == 1:
                                    plt.plot([combined_embeddings_[i, 0], combined_embeddings_[j, 0]], 
                                        [combined_embeddings_[i, 1], combined_embeddings_[j, 1]], 'k--')

                        plt.title(f'Embeddings at Epoch {epoch}')
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.legend()
                        plt.axis('equal')
                        plt.grid(True)
                        
                        # Plot 3: Early stopping info
                        if early_stopping:
                            plt.subplot(1, 3, 3)
                            plt.text(0.1, 0.8, f'Early Stopping: {"ON" if early_stopping else "OFF"}', 
                                fontsize=12, transform=plt.gca().transAxes)
                            plt.text(0.1, 0.7, f'Patience: {patience}', 
                                fontsize=10, transform=plt.gca().transAxes)
                            plt.text(0.1, 0.6, f'Min Delta: {min_delta}', 
                                fontsize=10, transform=plt.gca().transAxes)
                            plt.text(0.1, 0.5, f'Current Patience: {patience_counter}', 
                                fontsize=10, transform=plt.gca().transAxes)
                            plt.text(0.1, 0.4, f'Best Loss: {best_loss:.6f}', 
                                fontsize=10, transform=plt.gca().transAxes)
                            plt.text(0.1, 0.3, f'Current Loss: {loss:.6f}', 
                                fontsize=10, transform=plt.gca().transAxes)
                            plt.xlim(0, 1)
                            plt.ylim(0, 1)
                            plt.axis('off')
                            plt.title('Early Stopping Status')
                        
                        plt.tight_layout()
                        plt.show()
                        plt.close()
                
                break


        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(np.arange(S))

    # Restore best weights if early stopping was used and requested
    if early_stopping and restore_best_weights and best_embeddings is not None:
        model.embeddings.data = best_embeddings
        if verbose:
            print(f"Restored best weights from epoch {stopped_epoch - patience_counter}")

    model.normalize()

    
    
    return model.embeddings
