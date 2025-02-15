import json
import math
import itertools
import collections
import functools
import random
import heapq

def find_cycles(edge_list):
    cycles = []
    
    def findNewCycles(path):
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
    
    def invert(path):
        return rotate_to_smallest(path[::-1])
    
    def rotate_to_smallest(path):
        n = path.index(min(path))
        return path[n:] + path[:n]
    
    def isNew(path):
        return path not in cycles
    
    def visited(node, path):
        return node in path
    
    for edge in edge_list:
        for node in edge:
            findNewCycles([node])
    
    return cycles

def whether_triangulated(graph, adjacency_matrix):
    cycles = find_cycles(graph)
    results = [False for i in range(len(cycles))]
    # print(cycles)
    for cycle in cycles:
        if (len(cycle) < 4):
            results[cycles.index(cycle)] = True
            continue
        for i in range(len(cycle)):
           for j in range(i + 2, len(cycle)):
                # check if i,j are adjacent in the cycle
                if i == 0 and j == len(cycle) - 1:
                    continue
                if adjacency_matrix[cycle[i]][cycle[j]] == 1:
                    # print(cycle, i, j)
                    results[cycles.index(cycle)] = True
                    break
    if all(results):
        return True
    return False




########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################


def whether_triangulated(graph, adjacency_matrix):
    cycles = find_cycles(graph)
    results = [False for i in range(len(cycles))]
    # print(cycles)
    for cycle in cycles:
        if (len(cycle) < 4):
            results[cycles.index(cycle)] = True
            continue
        for i in range(len(cycle)):
           for j in range(i + 2, len(cycle)):
                # check if i,j are adjacent in the cycle
                if i == 0 and j == len(cycle) - 1:
                    continue
                if adjacency_matrix[cycle[i]][cycle[j]] == 1:
                    # print(cycle, i, j)
                    results[cycles.index(cycle)] = True
                    break
    if all(results):
        return True
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
        self.num_variables = data['VariablesCount']
        num_cliques = data['Potentials_count']
        # print(data)
        for i in range(num_cliques):
            self.cliques.append(data['Cliques and Potentials'][i]['cliques'])
        # print(self.cliques)
        self.adjacency_matrix = [[0 for i in range(self.num_variables)] for j in range(self.num_variables)]
        for i in self.cliques:
            for j in range(len(i)):
                for k in range(j + 1, len(i)):
                    self.adjacency_matrix[i[j]][i[k]] = 1
                    self.adjacency_matrix[i[k]][i[j]] = 1

        for i  in range(self.num_variables):
            for j in range(i+1, self.num_variables):
                if self.adjacency_matrix[i][j] == 1:
                    self.edges.append([i, j])
        
        for i in range(self.num_variables):
            temp = []
            for j in range(self.num_variables):
                if self.adjacency_matrix[i][j] == 1:
                    temp.append(j)
            self.adjacency_list.append(temp)
        for i in range(num_cliques):
            self.cliques.append(data['Cliques and Potentials'][i]['cliques'])
            self.potentials[tuple(data['Cliques and Potentials'][i]['cliques'])] = []
            for j in range(pow(2, data['Cliques and Potentials'][i]['clique_size'])):
                self.potentials[tuple(data['Cliques and Potentials'][i]['cliques'])].append(data['Cliques and Potentials'][i]['potentials'][j])
        self.k_value = data['k value (in top k)']

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
            if not P and not X:  # Maximal clique found
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
        if not whether_triangulated(self.edges, self.adjacency_matrix):
            
            temp_adj_list = [[j for j in i] for i in self.adjacency_list]

            degrees = {}

            vertices_left = set(range(self.num_variables))
            
            for i in vertices_left:
                degree = len(temp_adj_list[i])
                degrees[i] = degree

            while len(vertices_left) > 0:
                temp = min(degrees.values())
                vertex = [key for key in degrees if degrees[key] == temp][0]

                vertices_left.remove(vertex)
                degrees.pop(vertex)

                for i, j in itertools.combinations(temp_adj_list[vertex], 2):
                    if i not in temp_adj_list[j]:
                        self.adjacency_list[i].append(j)
                        self.adjacency_list[j].append(i)
                        temp_adj_list[i].append(j)
                        temp_adj_list[j].append(i)
                
                neighbours = [i for i in temp_adj_list[vertex]]

                for i in neighbours:
                    temp_adj_list[i].remove(vertex)               
                    degree = len(temp_adj_list[i])
                    degrees[i] = degree

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
        # maximal_cliques = [[1,2,3], [2,3,4], [3,4,5], [4,5,6]]
        for c1, c2 in itertools.combinations(maximal_cliques, 2):
            intersection_set = set(c1) & set(c2)
            if intersection_set :
                weight = len(intersection_set)
                heapq.heappush(tree_edges, (-weight, c1, c2))
        
        parent_child_map = {tuple(c): tuple(c) for c in maximal_cliques}
        rank = {tuple(c): 0 for c in maximal_cliques}

        def find_parent(c):
            if parent_child_map[c] != c:
                parent_child_map[c] = find_parent(parent_child_map[c])
            return parent_child_map[c]

        def union(c1, c2):
            root1, root2 = find_parent(c1), find_parent(c2)
            if root1 != root2:
                if rank[root1] > rank[root2]:
                    parent_child_map[root2] = root1
                elif rank[root2] > rank[root1]:
                    parent_child_map[root1] = root2
                else: 
                    parent_child_map[root2] = root1
                    rank[root1] += 1
                return True
            return False
        
        mst = []
        while tree_edges:
            weight, c1, c2 = heapq.heappop(tree_edges)
            if union(tuple(c1), tuple(c2)):
                mst.append([c1,c2])
        # print (mst)
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
        # print(self.maximal_cliques)
        maximal_cliques_copy = []
        for clique in self.maximal_cliques:
            maximal_cliques_copy.append(list(clique))
        self.maximal_cliques = maximal_cliques_copy
        for clique in self.maximal_cliques:
            cliques_subsumed = []
            for clique_subsumed in self.cliques:
                if set(clique).issubset(set(clique_subsumed)) and clique not in cliques_subsumed:
                    cliques_subsumed.append(clique_subsumed)
            potentials = [1 for i in range(pow(2, len(clique)))]
            for i in range(len(potentials)):
                assignment = []
                for j in range(len(clique)):
                    assignment.append((i >> j) & 1)
                for clique_subsumed in cliques_subsumed:
                    index = 0
                    for j in range(len(clique_subsumed)):
                        index = index * 2 + assignment[clique.index(clique_subsumed[j])]
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
        junction_tree_adj_list = {}
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
        
        root = tuple(self.maximal_cliques[0])
        depth_map = {root:0}

        def dfs(node, parent, depth):
            for child in junction_tree_adj_list[node]:
                if child != parent:
                    depth_map[child] = depth
                    dfs(child, node, depth + 1)
        
        dfs(root, None, 1)
        print("Depth Map:", depth_map)
        
        def send_message(from_clique, to_clique, parent_map, clique_potentials, messages):
            separator = tuple(sorted(set(from_clique) & set(to_clique)))
            message = [0] * (2 ** len(separator))
            from_potential = clique_potentials[from_clique][:] 
            for neighbor in parent_map.get(from_clique, []):  
                if neighbor != to_clique and (neighbor, from_clique) in messages:
                    incoming_message = messages[(neighbor, from_clique)]
                    for i in range(len(from_potential)):
                        assignment = [(i >> j) & 1 for j in range(len(from_clique))]
                        separator_index = sum([(assignment[from_clique.index(var)] << j) for j, var in enumerate(set(from_clique) & set(neighbor))])
                        from_potential[i] *= incoming_message[separator_index]
            for i in range(len(from_potential)):
                assignment = [(i >> j) & 1 for j in range(len(from_clique))]
                separator_index = sum([(assignment[from_clique.index(var)] << j) for j, var in enumerate(separator)])
                message[separator_index] += from_potential[i] 
            messages[(from_clique, to_clique)] = message

        messages = {}   
        clique_potentials = {tuple(clique): self.maximal_clique_potentials[i] for i, clique in enumerate(self.maximal_cliques)} 
        max_depth = max(depth_map.values()) 
        # print(max_depth)   
        for depth in range(max_depth, -1, -1):
            for clique, d in depth_map.items():
                if d == depth: 
                    parent = [p for p in junction_tree_adj_list[clique] if depth_map[p] == depth - 1]
                    for p in parent:
                        send_message(clique, p, junction_tree_adj_list, clique_potentials, messages)

        for depth in range(0, max_depth + 1): 
            for clique, d in depth_map.items():
                if d == depth: 
                    children = [c for c in junction_tree_adj_list[clique] if depth_map[c] == depth + 1]
                    for c in children:
                        send_message(clique, c, junction_tree_adj_list, clique_potentials, messages)

        # print(messages)
        # Downward phase: compute Z
        root_potential = clique_potentials[root]
        for neighbor in self.maximal_cliques:
            neighbor = tuple(neighbor)
            if set(root) & set(neighbor):
                if (neighbor, root) in messages:
                    message = messages[(neighbor, root)]
                    separator = tuple(sorted(set(root) & set(neighbor)))

                    for i in range(len(root_potential)):
                        assignment = [(i >> j) & 1 for j in range(len(root))]
                        separator_index = sum([(assignment[root.index(var)] << j) for j, var in enumerate(separator)])
                        root_potential[i] *= message[separator_index]  # Modify in-place

        Z = sum(root_potential)
        self.messages = messages
        self.z = Z
        # print(Z)
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
        clique_potentials = {tuple(clique): self.maximal_clique_potentials[i] for i, clique in enumerate(self.maximal_cliques)}
        z_value = self.get_z_value()
        junction_tree = self.get_junction_tree()
        adjacency_list = {}
        for i in range(len(self.maximal_cliques)):
            adjacency_list[tuple(self.maximal_cliques[i])] = []
        for i in range(len(junction_tree)):
            adjacency_list[tuple(junction_tree[i][0])].append(junction_tree[i][1])
            adjacency_list[tuple(junction_tree[i][1])].append(junction_tree[i][0])
        marginals = [[0,0] for _ in range(self.num_variables)]
        for i in range(self.num_variables):
            m_clique = []
            for clique in self.maximal_cliques:
                if i in clique:
                    m_clique = clique
                    break
            if m_clique == []:
                print("ERROR!!!")
                return
            for nc in self.maximal_cliques:
                if set(nc) & set(m_clique):
                    separator = tuple(sorted(set(m_clique) & set(nc)))
                    # message = [0] * (2 ** len(separator))
                    from_potential = clique_potentials[tuple(m_clique)][:] 
                    print((nc, m_clique), self.messages)
                    incoming_message = self.messages[(tuple(nc), tuple(m_clique))]
                    for ci in range(len(m_clique)):
                        assignment = [(ci >> j) & 1 for j in range(len(m_clique))]
                        separator_index = sum([(assignment[m_clique.index(var)] << j) for j, var in enumerate(set(m_clique) & set(nc))])
                        from_potential[ci] *= incoming_message[separator_index]
            ind = m_clique.index(i)
            for j in range(len(from_potential)):
                if (j & (1<<ind)):
                    marginals[1] += from_potential[j]
                else:
                    marginals[0] += from_potential[j]
            marginals[1] /= z_value
            marginals[0] /= z_value
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
        pass



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