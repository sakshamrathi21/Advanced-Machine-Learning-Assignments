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
        # print(clique_potentials)
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
        self.clique_potentials = copy.deepcopy(clique_potentials)
        # print("hello copy", self.clique_potentials)
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
        # print("Clique potentials:", self.maximal_clique_potentials)
        # print(self.maximal_clique_potentials)
        # self.maximal_clique_potentials = 
        # self.maximal_clique_potentials = dict(clique_potentials_copy)
        # print(self.maximal_clique_potentials)
        Z = sum(root_potential)
        self.messages = messages
        self.z = Z
        # print(Z)
        return Z