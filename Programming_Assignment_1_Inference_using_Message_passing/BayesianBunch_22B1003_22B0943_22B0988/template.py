import json
import itertools
import heapq
import copy
from itertools import product

def find_cycles(edge_list):
    """
    Takes as argument a list of edges in the graph and returns a list of cycles
    """
    cycles = []
    
    def findNewCycles(path):
        """
        Takes as argument a path in the graph and finds the cycles that can be formed with it
        """
        start_node = path[0]
        next_node = None
        
        for edge in edge_list:
            node1, node2 = edge
            if start_node in edge:
                next_node = node2 if node1 == start_node else node1
                if not visited(next_node, path):
                    findNewCycles([next_node] + path)
                elif len(path) > 2 and next_node == path[-1]:
                    p = rotate_to_smallest(path)
                    inv = invert(p)
                    if isNew(p) and isNew(inv):
                        cycles.append(p)
    
    def rotate_to_smallest(path):
        """
        Takes as argument a path in the graph and rotates it to the smallest node
        """
        n = path.index(min(path))
        return path[n:] + path[:n]
    
    def invert(path):
        """
        Takes as argument a path in the graph and returns its inverse,
        starting from the smallest node       
        """
        return rotate_to_smallest(path[::-1])
    
    def isNew(path):
        """
        Takes as argument a path in the graph and returns True
        if it is a new cycle, False otherwise
        """
        return path not in cycles
    
    def visited(node, path):
        """
        Takes as argument a node and a path in the graph and returns True
        if the node is in the path, False otherwise
        """
        return node in path
    
    for edge in edge_list:
        for node in edge:
            findNewCycles([node])
    
    return cycles


def whether_triangulated(graph, adjacency_matrix):
    """
    Takes as argument a graph and its adjacency matrix and returns True if
    the graph is triangulated, False otherwise
    """
    cycles = find_cycles(graph)     # finds all cycles in the graph
    results = [False for i in range(len(cycles))]
    for cycle in cycles:
        if (len(cycle) < 4):
            results[cycles.index(cycle)] = True     # if the cycle has less than 4 nodes, it is chordal
            continue
        # for cycles of length >= 4
        for i in range(len(cycle)):
           for j in range(i + 2, len(cycle)):
                if i == 0 and j == len(cycle) - 1:
                    continue
                if adjacency_matrix[cycle[i]][cycle[j]] == 1:
                    # there is a shortcut in the graph, so the cycle is chordal
                    results[cycles.index(cycle)] = True
                    break
                
    if all(results):    # all cycles are chordal; graph is triangulated
        return True
    # there is atleast one cycle that violates the triangulated property
    return False


class Inference:
    z = -1
    num_variables = 0
    cliques = []
    edges = []
    potentials = {}
    k_value = 0
    adjacency_matrix = [[]]
    adjacency_list = []
    maximal_clique_potentials = []
    maximal_cliques = []
    messages = {}
    clique_potentials = {}

    def clear(self):
        self.z = -1
        self.num_variables = 0
        self.cliques = []
        self.edges = []
        self.potentials = {}
        self.k_value = 0
        self.adjacency_matrix = [[]]
        self.adjacency_list = []
        self.maximal_clique_potentials = []
        self.maximal_cliques = []
        self.messages = {}
        self.clique_potentials = {}

    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """

        self.clear()
        self.num_variables = data['VariablesCount']
        self.k_value = data['k value (in top k)']

        num_cliques = data['Potentials_count']

        for i in range(num_cliques):
            if data['Cliques and Potentials'][i] in self.cliques:
                continue
            self.cliques.append(data['Cliques and Potentials'][i]['cliques'])

        # initialize adjacency matrix
        self.adjacency_matrix = [[0 for i in range(self.num_variables)] for j in range(self.num_variables)]

        for i in self.cliques:
            for j in range(len(i)):
                for k in range(j + 1, len(i)):
                    self.adjacency_matrix[i[j]][i[k]] = 1
                    self.adjacency_matrix[i[k]][i[j]] = 1

        # initialize edge list
        for i  in range(self.num_variables):
            for j in range(i+1, self.num_variables):
                if self.adjacency_matrix[i][j] == 1:
                    self.edges.append([i, j])
        
        # initialize adjacency list
        for i in range(self.num_variables):
            temp = []
            for j in range(self.num_variables):
                if self.adjacency_matrix[i][j] == 1:
                    temp.append(j)
            self.adjacency_list.append(temp)

        # initialize potentials for each clique
        for i in range(num_cliques):
            clique = tuple(data['Cliques and Potentials'][i]['cliques'])
            new_potentials = data['Cliques and Potentials'][i]['potentials']
            if clique in self.potentials:
                for j in range(pow(2, data['Cliques and Potentials'][i]['clique_size'])):
                    self.potentials[clique][j] *= new_potentials[j]
            else:
                self.potentials[clique] = []
                for j in range(pow(2, data['Cliques and Potentials'][i]['clique_size'])):
                    self.potentials[tuple(data['Cliques and Potentials'][i]['cliques'])].append(data['Cliques and Potentials'][i]['potentials'][j])
                    

    def get_maximal_cliques(self):
        """
        Extract maximal cliques from the triangulated graph using the Bron–Kerbosch algorithm.
        
        This function assumes the graph is already chordal.
        """

        def bron_kerbosch(R, P, X, cliques):
            """
            Recursive Bron–Kerbosch algorithm for finding maximal cliques.
            R: Current clique
            P: Potential nodes to be added
            X: Nodes that should not be added (avoid duplicates)
            cliques: List to store found cliques
            """
            if not P and not X:  
                cliques.append(R)
                return
            
            for v in list(P):
                bron_kerbosch(R | {v}, P & set(self.adjacency_list[v]), X & set(self.adjacency_list[v]), cliques)
                P.remove(v)
                X.add(v)

        cliques = []
        bron_kerbosch(set(), set(range(self.num_variables)), set(), cliques)
        return cliques


    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
        # if the graph is already triangulated, no need to triangulate
        if not whether_triangulated(self.edges, self.adjacency_matrix):
            # add edges to make the graph chordal

            temp_adj_list = [[j for j in i] for i in self.adjacency_list]   # deep copy that we can remove vertices from
            degrees = {}

            vertices_left = set(range(self.num_variables))  # set of vertices in the remaining untriangulated graph

            for i in vertices_left:
                degree = len(temp_adj_list[i])
                degrees[i] = degree

            while len(vertices_left) > 0:
                # we use the minimum degree heuristic to select the vertex that is to be made simplicial
                temp = min(degrees.values())
                vertex = [key for key in degrees if degrees[key] == temp][0]

                vertices_left.remove(vertex)
                degrees.pop(vertex)

                # make the vertex simplicial by adding edges between each pair of its neighbours
                for i, j in itertools.combinations(temp_adj_list[vertex], 2):
                    if i not in temp_adj_list[j]:
                        self.adjacency_list[i].append(j)
                        self.adjacency_list[j].append(i)
                        temp_adj_list[i].append(j)
                        temp_adj_list[j].append(i)
                
                neighbours = [i for i in temp_adj_list[vertex]]

                # remove the vertex from the adjacency list of its neighbours
                for i in neighbours:
                    temp_adj_list[i].remove(vertex)               
                    degree = len(temp_adj_list[i])
                    degrees[i] = degree

        # now we extract the maximal cliques from the triangulated graph
        self.maximal_cliques = self.get_maximal_cliques()       


    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        tree_edges = []
        maximal_cliques = self.maximal_cliques
        for c1, c2 in itertools.combinations(maximal_cliques, 2):
            intersection_set = set(c1) & set(c2)
            if intersection_set : # for all the pairs of cliques that have a set of common variables
                weight = len(intersection_set)
                heapq.heappush(tree_edges, (-weight, c1, c2)) # add negative of size of intersection to the heap to implement a max heap
        
        parent_child_map = {tuple(c): tuple(c) for c in maximal_cliques} # Each clique is its own parent initially
        rank = {tuple(c): 0 for c in maximal_cliques} # Initial rank is zero

        def find_parent(c):
            if parent_child_map[c] != c:
                parent_child_map[c] = find_parent(parent_child_map[c])
            return parent_child_map[c]

        def union(c1, c2): # merges two cliques into the same set using union by rank
            root1, root2 = find_parent(c1), find_parent(c2)
            if root1 != root2:  # only merge if they belong to different sets
                if rank[root1] > rank[root2]: # attach smaller tree to larger tree
                    parent_child_map[root2] = root1
                elif rank[root2] > rank[root1]:
                    parent_child_map[root1] = root2
                else: 
                    parent_child_map[root2] = root1
                    rank[root1] += 1
                return True
            return False
        
         # Minimum Spanning Tree construction using Kruskal's algorithm
        mst = []
        while tree_edges:
            weight, c1, c2 = heapq.heappop(tree_edges) # extract the edge with highest weight
            if union(tuple(c1), tuple(c2)): # if adding this edge does not create a cycle
                mst.append([c1,c2]) # add the edge to the junction tree
        return mst
        

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """

        maximal_cliques_copy = []   # deep copy of maximal_cliques
        for clique in self.maximal_cliques:
            maximal_cliques_copy.append(list(clique))

        self.maximal_cliques = maximal_cliques_copy
        
        cliques_taken_so_far = []

        for clique in self.maximal_cliques:
            
            cliques_subsumed = []   # holds the list of subcliques whose potentials have to be multiplied to get the potential of the maximal clique
            for clique_subsumed in self.cliques:
                if set(clique_subsumed).issubset(set(clique)) and clique_subsumed not in cliques_subsumed and clique_subsumed not in cliques_taken_so_far:
                    cliques_subsumed.append(clique_subsumed)
                    cliques_taken_so_far.append(clique_subsumed)

            potentials = [1 for _ in range(pow(2, len(clique)))]    # potentials for each possible assignment of variables in the clique

            for i in range(len(potentials)):
                assignment = []
                for j in range(len(clique)):
                    assignment.append((i >> j) & 1)
                for clique_subsumed in cliques_subsumed:
                    index = 0
                    for j in range(len(clique_subsumed)):
                        index = index * 2 + assignment[clique.index(clique_subsumed[j])]
                    # multiply the potentials of the subcliques to get the potential of the maximal clique
                    potentials[i] *= self.potentials[tuple(clique_subsumed)][index] 
                    
            self.maximal_clique_potentials.append(potentials)
                

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """
        if self.z != -1:
            return self.z
        junction_tree = self.get_junction_tree() 
        junction_tree_adj_list = {} # create adjacency list for the junction tree
        for edge in junction_tree:
            a = tuple(edge[0])
            b = tuple(edge[1])
            if a not in junction_tree_adj_list:
                junction_tree_adj_list[a] = [b]
            else:
                junction_tree_adj_list[a].append(b)
            if b not in junction_tree_adj_list:
                junction_tree_adj_list[b] = [a]
            else:
                junction_tree_adj_list[b].append(a)
        
        root = tuple(self.maximal_cliques[0]) # declare one of the maximal_cliques as root
        depth_map = {root:0} # depth map to map each node in the junction tree to its depth from root

        def dfs(node, parent, depth):
            for child in junction_tree_adj_list[node]:
                if child != parent:
                    depth_map[child] = depth
                    dfs(child, node, depth + 1)
        
        dfs(root, None, 1)
        
        def send_message(from_clique, to_clique, parent_map, clique_potentials, messages):
            separator = tuple(sorted(set(from_clique) & set(to_clique)))  # set of variables in the intersection of sending and receiving node
            message = [0] * (2 ** len(separator)) # Initialize message to be sent
            from_potential = clique_potentials[from_clique][:] 
            # Consider messages received from other neighboring cliques
            for neighbor in parent_map.get(from_clique, []):  
                if neighbor != to_clique and (neighbor, from_clique) in messages:
                    incoming_message = messages[(neighbor, from_clique)]
                    for i in range(len(from_potential)):
                        # Compute the index for separator potential using the assignment of shared variables
                        assignment = [(i >> j) & 1 for j in range(len(from_clique))]
                        separator_index = sum([(assignment[from_clique.index(var)] << j) for j, var in enumerate(set(from_clique) & set(neighbor))])
                        # Multiply the potential with the incoming message value for the corresponding assignment
                        from_potential[i] *= incoming_message[separator_index]
            for i in range(len(from_potential)):
                assignment = [(i >> j) & 1 for j in range(len(from_clique))]
                separator_index = sum([(assignment[from_clique.index(var)] << j) for j, var in enumerate(separator)])
                message[separator_index] += from_potential[i] 
            # Store the computed message
            messages[(from_clique, to_clique)] = message

        messages = {}   
        clique_potentials = {tuple(clique): self.maximal_clique_potentials[i] for i, clique in enumerate(self.maximal_cliques)} 
        max_depth = max(depth_map.values())   # maximum depth of the junction tree
        
        # Perform message passing from leaf nodes to the root
        for depth in range(max_depth, -1, -1):
            for clique, d in depth_map.items():
                if d == depth: 
                    parent = [p for p in junction_tree_adj_list[clique] if depth_map[p] == depth - 1]
                    for p in parent:
                        send_message(clique, p, junction_tree_adj_list, clique_potentials, messages)
                        
        # Perform message passing from in the opposite direction 
        for depth in range(0, max_depth + 1): 
            for clique, d in depth_map.items():
                if d == depth: 
                    children = [c for c in junction_tree_adj_list[clique] if depth_map[c] == depth + 1]
                    for c in children:
                        send_message(clique, c, junction_tree_adj_list, clique_potentials, messages)
                        
        self.clique_potentials = copy.deepcopy(clique_potentials)
        root_potential = clique_potentials[root]
        for neighbor in self.maximal_cliques:
            neighbor = tuple(neighbor)
            # if intersection of root and neighbour is not null
            if set(root) & set(neighbor):
                # If a message exists from neighbor to root, process it
                if (neighbor, root) in messages:
                    message = messages[(neighbor, root)]
                    separator = tuple(sorted(set(root) & set(neighbor)))

                    for i in range(len(root_potential)):
                        assignment = [(i >> j) & 1 for j in range(len(root))]
                        separator_index = sum([(assignment[root.index(var)] << j) for j, var in enumerate(separator)])
                         # Multiply the root potential value with the corresponding message value
                        root_potential[i] *= message[separator_index] 
        # compute z by summing over all values in the root potential
        Z = sum(root_potential)
        self.messages = messages
        self.z = Z
        return Z

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
        Refer to the sample test case for the expected format of the marginals.
        """
        clique_potentials = self.clique_potentials
        z_value = self.get_z_value()
        junction_tree = self.get_junction_tree()

        # Build an adjacency list representation of the junction tree
        adjacency_list = {}
        for i in range(len(self.maximal_cliques)):
            adjacency_list[tuple(self.maximal_cliques[i])] = []

        for i in range(len(junction_tree)):
            adjacency_list[tuple(junction_tree[i][0])].append(junction_tree[i][1])
            adjacency_list[tuple(junction_tree[i][1])].append(junction_tree[i][0])

        # Initialize marginals for each variable (each has two probabilities: P(X=0) and P(X=1))
        marginals = [[0,0] for _ in range(self.num_variables)]

        # Compute marginals for each variable
        for i in range(self.num_variables):
            m_clique = []

            # Find the first maximal clique that contains variable i
            for clique in self.maximal_cliques:
                if i in clique:
                    m_clique = clique
                    break

            # If no clique contains the variable, print an error
            if m_clique == []:
                print("ERROR!!!")
                return
            
            # Start with the potential of the clique containing the variable
            from_potential = clique_potentials[tuple(m_clique)][:] 

            # Incorporate incoming messages from neighboring cliques
            for nc in self.maximal_cliques:
                if set(nc) & set(m_clique) and nc != m_clique and (tuple(nc), tuple(m_clique)) in self.messages:
                    # Compute the separator set (shared variables between cliques)
                    separator = tuple(sorted(set(m_clique) & set(nc)))
                    incoming_message = self.messages[(tuple(nc), tuple(m_clique))]

                    # Adjust the clique potential using the incoming message
                    for ci in range(len(from_potential)):
                        assignment = [(ci >> j) & 1 for j in range(len(m_clique))]
                        separator_index = sum([(assignment[m_clique.index(var)] << j) for j, var in enumerate(set(m_clique) & set(nc))])
                        from_potential[ci] *= incoming_message[separator_index]

            ind = m_clique.index(i)

            # Sum over the relevant entries to compute marginal probabilities
            for j in range(len(from_potential)):
                if (j & (1<<ind)):
                    marginals[i][1] += from_potential[j]
                else:
                    marginals[i][0] += from_potential[j]

            # Normalize the probabilities using the partition function
            marginals[i][1] /= z_value
            marginals[i][0] /= z_value
            
        return marginals

    def compute_top_k(self):
        """ 
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """
        junction_tree = self.get_junction_tree() 
        junction_tree_adj_list = {}
        for edge in junction_tree: # create adjacency list for the junction tree
            a = tuple(edge[0])
            b = tuple(edge[1])
            if a not in junction_tree_adj_list:
                junction_tree_adj_list[a] = [b]
            else:
                junction_tree_adj_list[a].append(b)
            if b not in junction_tree_adj_list:
                junction_tree_adj_list[b] = [a]
            else:
                junction_tree_adj_list[b].append(a)
        root = tuple(self.maximal_cliques[0])  # declare one of the maximal_cliques as root
        depth_map = {root:0}   # depth map to map each node in the junction tree to its depth from root
        def dfs(node, parent, depth):
            for child in junction_tree_adj_list[node]:
                if child != parent:
                    depth_map[child] = depth
                    dfs(child, node, depth + 1)
        dfs(root, None, 1)
        
        def send_message(from_clique, to_clique, parent_map, clique_potentials, messages):
            set_neighbor_variables = set()
            variables_seen = set(from_clique)  # current set of variables encountered in the process of joint probability computation
            # Consider messages received from other neighboring cliques
            for neighbor in parent_map.get(from_clique, []): 
                if neighbor != to_clique and (neighbor, from_clique) in messages:
                    variables_seen = variables_seen.union(set(messages[(neighbor, from_clique)][0]))
                    set_neighbor_variables = set_neighbor_variables.union(set(messages[(neighbor, from_clique)][0]))
                    
            # Initialize message to be sent
            message_to_send = (variables_seen, {})
            for i in range(2 ** len(variables_seen)):
                message_to_send[1][i] = 1
            # message_to_send = (variables_seen, [1] * (2 ** len(variables_seen)))
            list_variables_seen = list(variables_seen)  
            from_potential = clique_potentials[from_clique][:]
            # Compute message values for each assignment
            for i in range(len(message_to_send[1])):
                from_potential_index = 0
                assignment = [(i >> j) & 1 for j in range(len(variables_seen))]  # Convert to binary assignment

                # Map assignment to clique potential index
                for j in range(len(from_clique)):
                    from_potential_index = from_potential_index * 2 + assignment[list_variables_seen.index(from_clique[(len(from_clique) - 1) - j])]  

                # Multiply with the corresponding clique potential value
                message_to_send[1][i] *= from_potential[from_potential_index]
                # Multiply with messages received from other neighboring cliques
                for neighbor in parent_map.get(from_clique, []):
                    if neighbor != to_clique and (neighbor, from_clique) in messages:
                        incoming_message = messages[(neighbor, from_clique)]
                        incoming_message_index = 0
                        for j in range(len(incoming_message[0])):
                            incoming_message_index = incoming_message_index * 2 + assignment[list_variables_seen.index(list(incoming_message[0])[len(incoming_message[0]) - 1 - j])]
                        if incoming_message_index in incoming_message[1]:
                            message_to_send[1][i] *= incoming_message[1][incoming_message_index]
            # Store the computed message
            if to_clique is not None:
                variables_extra = set(from_clique) - set(set_neighbor_variables)
            else:
                variables_extra = set(from_clique)
            list_neighboring_variables = list(set_neighbor_variables)
            if variables_extra != set():
                final_message = {}
                for i in range(pow(2, len(variables_extra))):
                    assignment_potential_pairs ={}
                    for j in range(pow(2, len(list_neighboring_variables))):
                        assignment = [0] * len(variables_seen)
                        index = 0
                        for k in range(len(variables_seen)):
                            if list(variables_seen)[k] in list_neighboring_variables:
                                assignment[k] = (j >> list_neighboring_variables.index(list(variables_seen)[k])) & 1
                            else:
                                assignment[k] = (i >> list(variables_extra).index(list(variables_seen)[k])) & 1
                        for k in range(len(variables_seen)):
                            index = index * 2 + assignment[len(variables_seen) - 1 - k]
                        assignment_potential_pairs[index] = message_to_send[1][index]
                    assignment_prob_pairs = list(assignment_potential_pairs.items())
                    assignment_prob_pairs.sort(key=lambda x: -x[1])
                    top_k_assignments = {}
                    for assignment, prob in assignment_prob_pairs:
                        if len(top_k_assignments) < self.k_value:
                            top_k_assignments[assignment] = prob
                            final_message[assignment] = prob
                message_to_send = (variables_seen, final_message)
                    

                    




            # print("Extra variables", variables_extra)
            if pow(2, len(variables_extra)) < self.k_value*pow(2, len(variables_seen)):
                messages[(from_clique, to_clique)] = message_to_send
            else:
                # Sort the assignments by probability in descending order
                assignment_prob_pairs = list(zip(list(product([0, 1], repeat=len(variables_extra))), message_to_send[1]))
                assignment_prob_pairs.sort(key=lambda x: -x[1])
                top_k_variable_extra_assignments = []
                final_assignments = {}
                for assignment, prob in assignment_prob_pairs:
                    # first we need to find the assignment corresponding to the variables_extra
                    new_assignment= [0] * len(variables_extra)
                    for i in range(len(variables_extra)):
                        new_assignment[i] = assignment[list(variables_extra).index(list(variables_extra)[i])]
                    if new_assignment not in top_k_variable_extra_assignments and len(top_k_variable_extra_assignments) < self.k_value:
                        top_k_variable_extra_assignments.append((tuple(new_assignment), prob))
                    if new_assignment in top_k_variable_extra_assignments:
                        final_assignments[assignment] = prob
                if len(top_k_variable_extra_assignments) < self.k_value:
                    print("ERROR")
                messages[(from_clique, to_clique)] = (variables_seen, final_assignments)
                
                

        messages = {}
        clique_potentials = self.clique_potentials
        max_depth = max(depth_map.values()) # maximum depth of the junction tree

        # Perform message passing from leaf nodes to the root
        for depth in range(max_depth, -1, -1):
            for clique, d in depth_map.items():
                if d == depth:
                    # Get parent cliques (adjacent cliques at a lower depth value)
                    parent = [p for p in junction_tree_adj_list[clique] if depth_map[p] == depth - 1]
                    for p in parent:
                        send_message(clique, p, junction_tree_adj_list, clique_potentials, messages)
        # Send final message from root to compute marginal probabilities
        send_message(root, None, junction_tree_adj_list, clique_potentials, messages)
        message_final = messages[(root, None)]
        message_final_adapted = (message_final[0], [0]*pow(2, len(message_final[0])))
        for assignment, prob in message_final[1].items():
            message_final_adapted[1][assignment] = prob
        message_final = message_final_adapted
        # Compute normalized probabilities
        assignments = list(product([0, 1], repeat=self.num_variables)) # All possible binary assignments
        probabilities = message_final[1]
        for i in range(len(probabilities)):
            probabilities[i] /= self.z # Normalize using partition function z
    
        assignment_prob_pairs = list(zip(assignments, probabilities))
        assignment_prob_pairs.sort(key=lambda x: -x[1]) # Sort assignments by probability in descending order
        new_ass = []
        for assignment, prob in assignment_prob_pairs:
            new_ass.append((tuple(reversed(assignment)), prob))
        return new_ass[:self.k_value] # Return top k assignments
        
        self.get_z_value()
        num_vars = self.num_variables
        assignments = list(product([0, 1], repeat=num_vars))  
        prob_heap = []
        for assignment in assignments:
            prob = 1.0
            for clique, potential in self.clique_potentials.items():
                index = sum([(assignment[clique[i]] << i) for i in range(len(clique))])
                prob *= potential[index]
            prob /= self.z
            if len(prob_heap) < self.k_value:
                heapq.heappush(prob_heap, (prob, assignment))
            else:
                heapq.heappushpop(prob_heap, (prob, assignment))
        
        top_k_assignments = sorted(prob_heap, key=lambda x: -x[0])  
        return [(list(assignment), prob) for prob, assignment in top_k_assignments]
        



########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')