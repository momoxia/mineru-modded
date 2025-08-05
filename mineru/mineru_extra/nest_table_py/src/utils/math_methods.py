from itertools import combinations
from dataclasses import dataclass

# DFS 聚簇
def dfs_search(H, V):
        graph = {}
        
        for h in H:
            node = ("H", h)
            graph[node] = []
        for v in V:
            node = ("V", v)
            graph[node] = []

        for h, info in H.items():
            h_node = ("H", h)
            for v_ref in info["r"]:
                v_node = ("V", v_ref)
                if v_node in graph:
                    graph[h_node].append(v_node)
                    graph[v_node].append(h_node)
        
        for v, info in V.items():
            v_node = ("V", v)
            for h_ref in info["r"]:
                h_node = ("H", h_ref)
                if h_node in graph:
                    graph[v_node].append(h_node)
                    graph[h_node].append(v_node)
        
        visited = set()
        classes = []
        def dfs(node, component):
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
        
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                classes.append(component)
        return classes

# 获取组合
def permutations(l: list):
        pairs = combinations(l, 2)
        return [tuple(sorted(pair)) for pair in pairs]

# 获取交点
def intersections(lines):
        horizontals = []  
        verticals = []  
   
        for line in lines:
            x, y, w, h = line
            if w > h:  
                y_center = y + h / 2.0  

                horizontals.append((y_center, x, x + w))
            else: 
                x_center = x + w / 2.0
                verticals.append((x_center, y, y + h))
    
        intersections = []

        for h_line in horizontals:
            h_y, h_x1, h_x2 = h_line
            for v_line in verticals:
                v_x, v_y1, v_y2 = v_line
                if (h_x1 <= v_x <= h_x2) and (v_y1 <= h_y <= v_y2):
                    intersections.append((v_x, h_y))
    
        intersections.sort(key=lambda point: (point[1], point[0]))
        output = []
        if len(intersections) == 4:
            output.append(intersections[0][0]) 
            output.append(intersections[0][1])
            output.append(intersections[3][0])
            output.append(intersections[3][1])
            if output == []:
                return None
            else:
                return (output[0], 
                        output[1],
                        output[2], 
                        output[3])
        else:
            return None