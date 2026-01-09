import random, numpy as np
from annealer import Annealer
import networkx as nx

class SymmetryAproximator(Annealer):
    def __init__(self, state, A, B, mfp, centrality):
        self.N, self.mfp, self.lfp, self.fp = A.shape[0], mfp, 0, 0
        self.A = self.B = A
        self.centrality = centrality
        if B is not None:
            self.B = B
        self.iNeighbor, self.dNeighbor = [], [set() for _ in range(self.N)]
        for i in range(self.N):
            neigh = set()
            for j in range(self.N):
                if A[j,i] == 1 and i != j:
                    neigh.add(j)
            self.iNeighbor.append(neigh)

        for i, s in enumerate(state):
            neigh = []
            for j in range(self.N):
                if self.B[j,i] == 1:
                    neigh.append(j)
                    self.dNeighbor[s].add(state[j])
                    
        
        # A dictionary of optimal constants used for annealing guided by different centralities
        # The keys are the names of the centralities, the values are tuples of the form (division_constant, probability_constant)
        self.constants_for_centralities = {
                'degree': (1, 0.01), #TODO: tune this + complex
                'closeness': (1, 0.01),
                'eigenvector': (1, 0.01),
                'betweenness': (1, 0.01),
                'pagerank': (1, 0.01),
                'clustering': (1, 0.01),
                }

        if self.centrality != 'unguided': 
            self.similarity_matrix = self.compute_similarity_matrix(self.A, centrality=self.centrality)

        super(SymmetryAproximator, self).__init__(state)  # important!


    def compute_similarity_matrix(self, A, centrality):
        """Input: A - adjacency matrix of a graph. Output: a similarity matrix based on the given centrality of the graph.
        """
        
        G = nx.from_numpy_array(A)
        
        # Choose the division constant based on the centrality, if it is known. Otherwise, use the default value of 1.
        division_constant = self.constants_for_centralities[centrality][0]
        
        match centrality:
            case 'degree':
                centrality_values = nx.degree_centrality(G)    
            case 'closeness':
                centrality_values = nx.closeness_centrality(G)
            case 'eigenvector':
                centrality_values = nx.eigenvector_centrality(G, max_iter=1000)
            case 'betweenness':
                centrality_values = nx.betweenness_centrality(G)
            case 'pagerank':
                centrality_values = nx.pagerank(G, alpha=0.85)
            case 'clustering':
                centrality_values = nx.clustering(G)
            case _:
                centrality_values = nx.degree_centrality(G)
          
        
        # Fill the difference matrix with absolute differences of centralities
        diff_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                diff_matrix[i, j] = abs(centrality_values[i] - centrality_values[j])
        
        # compute the inverse of the distance matrix - create a form of similariy measure.
        # Add a constant to avoid division by zero. The higher the constant, the more even the choices will be
        similarity_matrix = 1./(division_constant + diff_matrix)
        
        return similarity_matrix


    
    # compute dE for vertex swap
    def diff(self, a, aN, b, bN):
        c = len(self.iNeighbor[a].symmetric_difference(aN))
        d = len(self.iNeighbor[b].symmetric_difference(bN))
        return c+d


    # compute 1/4 ||A - pAp^T|| for given p, A
    def energy(self):
        n, m = self.A.shape[0], self.A.shape[1]
        B = self.B[:, self.state]
        diff = 0
        for r in range(n):
            v = self.state[r]
            for c in range(m):
                diff += abs(B[v,c] - self.A[r,c])
        return diff/4
    

    def rewire(self, a, b, reset):
        if reset:
            self.fp = self.lfp
        ida, idb = self.state[a], self.state[b]
        # check whether a, b are neighbors
        neighbors = idb in self.dNeighbor[ida]
        # delete for everyone
        for i in range(self.N):
            if ida in self.dNeighbor[i]:
                self.dNeighbor[i].remove(ida)
            if idb in self.dNeighbor[i]:
                self.dNeighbor[i].remove(idb)

        # add to new neighborhoods
        for n in self.dNeighbor[ida]:
            self.dNeighbor[n].add(idb)
        for n in self.dNeighbor[idb]:
            self.dNeighbor[n].add(ida)

        # fix swapped vertices
        self.dNeighbor[ida], self.dNeighbor[idb] = \
            self.dNeighbor[idb], self.dNeighbor[ida]

        if ida in self.dNeighbor[ida]:
            self.dNeighbor[ida].remove(ida)
        if idb in self.dNeighbor[idb]:
            self.dNeighbor[idb].remove(idb)
        if neighbors:
            self.dNeighbor[ida].add(idb)
            self.dNeighbor[idb].add(ida)

    def check_fp(self, a, b):
        temp_fp_var = self.fp
        if a == b:
            return False
        if self.state[a] == a:
            temp_fp_var -= 1
        if self.state[b] == b:
            temp_fp_var -= 1
        if self.state[a] == b:
            temp_fp_var += 1
        if self.state[b] == a:
            temp_fp_var += 1
        if temp_fp_var > self.mfp:
            return False
        self.lfp = self.fp
        self.fp = temp_fp_var
        return True
        

    def guided_move(self):
        """Move to the next state, swapping similar vertices based on the similarity matrix."""
        probability_constant = self.constants_for_centralities[self.centrality][1]
        a = random.randint(0, len(self.state) - 1) #random choice works significantly better then a choice based on how "bad" the vertex and its image are
        
        
        # choose the vertex to swap images with:


        # list of energy differences for each possible swap. Start with zeros
        energy_diffs = np.zeros(len(self.state))
        image_a = self.state[a] 
        
        sim_a = self.similarity_matrix.item(a,image_a) # similarity of a -> image of a
        
        for b in range(len(self.state)):
            image_b = self.state[b]

            # if the swap would create a fixed point, let the energy difference stay at 0 (it will be normalized later)
            if (not (a == image_b or b == image_a or a == b)):
                sim_b = self.similarity_matrix.item(b,image_b) # similarity of b -> image of b
            
                sim_a_new = self.similarity_matrix.item(a,image_b) # similarity of a -> image of b
                sim_b_new = self.similarity_matrix.item(b,image_a) # similarity of b -> image of a
           
            
                # Calculate energy difference. We want this value to be as large as possible
                energy_diff = sim_a_new + sim_b_new - sim_a - sim_b
                energy_diffs[b] = energy_diff
                
        # choose the second vertex b that will swap images with a using the energy differences as a probability distribution.
        # probability constant is used to avoid division by zero. Also, the higher it is, the more even the choices will be
        energy_diffs = [max(probability_constant, energy_diff) for energy_diff in energy_diffs]
        # Normalize to create a probability distribution
        energy_diffs_probs = np.array(energy_diffs) / sum(energy_diffs)
        
        # Choose b based on energy differences as probability distribution
        b = np.random.choice(range(len(self.state)), p = energy_diffs_probs)
            
        if self.check_fp(a,b):
            ida, idb = self.state[a], self.state[b]
            aN, bN = self.dNeighbor[ida], self.dNeighbor[idb]
            
            # compute initial energy
            initial = self.diff(ida, aN, idb, bN) 
            self.rewire(a,b,False)
            # update permutation
            self.state[a], self.state[b] = self.state[b], self.state[a]
            aN, bN = self.dNeighbor[ida], self.dNeighbor[idb]
            
            # compute new energy
            after = self.diff(ida, aN, idb, bN)
            return (after-initial)/2, a, b
        return 0, a, b
     
    def unguided_move(self):
        """Move to the next state, swapping vertices randomly."""
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        
        # enforce trace(P) = 0
        # if self.state[a] != b and self.state[b] != a and a != b:
        if self.check_fp(a,b):
            # compute initial energy
            ida, idb = self.state[a], self.state[b]
            aN, bN = self.dNeighbor[ida], self.dNeighbor[idb]
            initial = self.diff(ida, aN, idb, bN) 
            self.rewire(a,b,False)
            # update permutation
            self.state[a], self.state[b] = self.state[b], self.state[a]
            aN, bN = self.dNeighbor[ida], self.dNeighbor[idb]
            after = self.diff(ida, aN, idb, bN)

            return (after-initial)/2, a, b

        return 0, a, b

    def move(self):
        """Wrapper for the move function. If the centrality is unguided, move randomly. Otherwise, move based on the similarity matrix."""
        if self.centrality == 'unguided': return self.unguided_move()
        else: return self.guided_move()

def check(perm):
    """Check if a permutation has a fixed point."""
    for i, v in enumerate(perm):
        if i == v:
            return True
    return False

def annealing(a, b=None, temp=1, steps=30000, runs=1, fp=100, centrality='unguided'):
    best_state, best_energy = None, None
    N = len(a)
    for _ in range(runs): 
        perm = np.random.permutation(N)
        # start on a permutation without fixed points (to avoid anomalies)
        while check(perm):
            perm = np.random.permutation(N)
        SA = SymmetryAproximator(list(perm), a, b, fp, centrality=centrality)
        SA.Tmax = temp
        SA.Tmin = 0.01
        SA.steps = steps
        SA.copy_strategy = 'slice'
        state, e = SA.anneal()
        if best_energy == None or e < best_energy:
            best_state, best_energy = state, e
    return best_state, best_energy/(N*(N-1))*4