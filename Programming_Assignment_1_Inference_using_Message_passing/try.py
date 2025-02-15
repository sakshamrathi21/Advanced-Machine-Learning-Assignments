def compute_top_k(self):
        """ 
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """
        print(self.k_value)
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
        depth_map = {root: 0}
        # print(depth_map)
        def dfs(node, parent, depth):
            for child in junction_tree_adj_list[node]:
                if child != parent:
                    depth_map[child] = depth
                    dfs(child, node, depth + 1)
        
        dfs(root, None, 1)
        
        def send_message_top_k(from_clique, to_clique, parent_map, clique_potentials, messages, k):
            separator = tuple(sorted(set(from_clique) & set(to_clique)))
            message = []
            from_potential = list(enumerate(clique_potentials[from_clique][:]))             
            for neighbor in parent_map.get(from_clique, []): 
                # print("Neighbor", neighbor) 
                if neighbor != to_clique and (neighbor, from_clique) in messages:
                    incoming_messages = messages[(neighbor, from_clique)]
                    new_potentials = []
                    
                    for index, value in from_potential:
                        assignment = [(index >> j) & 1 for j in range(len(from_clique))]
                        separator_index = sum([(assignment[from_clique.index(var)] << j) for j, var in enumerate(set(from_clique) & set(neighbor))])
                        new_potentials.append((index, value * incoming_messages[separator_index][1]))
                    print("New potentials", new_potentials) 
                    from_potential = sorted(new_potentials, key=lambda x: -x[1])[:k]  # Keep only top-k values
            # print("hello", from_potential)
            separator_potentials = {}
            
            for index, value in from_potential:
                assignment = [(index >> j) & 1 for j in range(len(from_clique))]
                separator_index = sum([(assignment[from_clique.index(var)] << j) for j, var in enumerate(separator)])
                
                if separator_index not in separator_potentials:
                    separator_potentials[separator_index] = []
                separator_potentials[separator_index].append((index, value))
            # print("Check", separator_potentials)
            for sep_idx, values in separator_potentials.items():
                values.sort(key=lambda x: -x[1])
                message.append((sep_idx, sum(v[1] for v in values[:k])))  # Sum top-k contributions
            print("Message", message, from_clique, to_clique)
            messages[(from_clique, to_clique)] = message
        
        messages = {}
        clique_potentials = {tuple(clique): self.maximal_clique_potentials[i] for i, clique in enumerate(self.maximal_cliques)}
        max_depth = max(depth_map.values())
        
        for depth in range(max_depth, -1, -1):  # Send messages from leaves to root
            for clique, d in depth_map.items():
                if d == depth:
                    parent = [p for p in junction_tree_adj_list[clique] if depth_map[p] == depth - 1]
                    for p in parent:
                        send_message_top_k(clique, p, junction_tree_adj_list, clique_potentials, messages, self.k_value)
       
        root_potential = clique_potentials[root]
        for child in junction_tree_adj_list[root]:
            if (child, root) in messages:
                incoming_message = messages[(child, root)]
                for i in range(len(root_potential)):
                    assignment = [(i >> j) & 1 for j in range(len(root))]
                    separator_index = sum([(assignment[root.index(var)] << j) for j, var in enumerate(set(root) & set(child))])
                    if separator_index < len(incoming_message):
                        print(root_potential[i], "hello", incoming_message[separator_index])
                        root_potential[i] *= incoming_message[separator_index][1]
        print("Root potential", root_potential)
        top_k_assignments = sorted(enumerate(root_potential), key=lambda x: -x[1])[:self.k_value]  # Select top-k assignments at root
        print(top_k_assignments)
        return [(bin(index)[2:].zfill(len(root)), prob) for index, prob in top_k_assignments]