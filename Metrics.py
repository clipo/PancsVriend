import numpy as np

def calculate_all_metrics(grid):
    clusters = count_clusters(grid)
    switch_rate = compute_switch_rate(grid)
    distance = compute_distance(grid)
    mix_dev = compute_mix_deviation(grid)
    share = compute_share(grid)
    ghetto_rate = compute_ghetto_rate(grid)
    return {
        "clusters": clusters,
        "switch_rate": switch_rate,
        "distance": distance,
        "mix_deviation": mix_dev,
        "share": share,
        "ghetto_rate": ghetto_rate
    }

def count_clusters(grid):
    visited = np.zeros(grid.shape, dtype=bool)
    clusters = 0

    def dfs(r, c, type_id):
        stack = [(r, c)]
        while stack:
            r0, c0 = stack.pop()
            if visited[r0][c0]:
                continue
            visited[r0][c0] = True
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                r1, c1 = r0 + dr, c0 + dc
                if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                    agent = grid[r1][c1]
                    if agent and agent.type_id == type_id and not visited[r1][c1]:
                        stack.append((r1, c1))

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent and not visited[r][c]:
                dfs(r, c, agent.type_id)
                clusters += 1
    return clusters

def compute_switch_rate(grid):
    switches, total = 0, 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                types = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    r1, c1 = r + dr, c + dc
                    if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                        n = grid[r1][c1]
                        if n:
                            types.append(n.type_id)
                if len(types) > 1:
                    total += len(types) - 1
                    switches += sum(1 for i in range(len(types)-1) if types[i] != types[i+1])
    return switches / total if total > 0 else 0

def compute_distance(grid):
    from scipy.ndimage import distance_transform_edt
    type_mask = np.zeros(grid.shape, dtype=int)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                type_mask[r][c] = agent.type_id + 1
    dists = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                other_type = 1 if agent.type_id == 0 else 2
                other_mask = (type_mask == other_type)
                dist = distance_transform_edt(~other_mask)[r][c]
                dists.append(dist)
    return np.mean(dists) if dists else 0

def compute_mix_deviation(grid):
    deviations = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                like, unlike = 0, 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r1, c1 = r + dr, c + dc
                        if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                            n = grid[r1][c1]
                            if n:
                                if n.type_id == agent.type_id:
                                    like += 1
                                else:
                                    unlike += 1
                total = like + unlike
                if total > 0:
                    deviation = abs(0.5 - like / total)
                    deviations.append(deviation)
    return np.mean(deviations) if deviations else 0

def compute_share(grid):
    like, unlike = 0, 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r1, c1 = r + dr, c + dc
                        if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                            n = grid[r1][c1]
                            if n:
                                if n.type_id == agent.type_id:
                                    like += 1
                                else:
                                    unlike += 1
    total = like + unlike
    return like / total if total > 0 else 0

def compute_ghetto_rate(grid):
    ghettos = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                has_unlike = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r1, c1 = r + dr, c + dc
                        if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                            n = grid[r1][c1]
                            if n and n.type_id != agent.type_id:
                                has_unlike = True
                                break
                if not has_unlike:
                    ghettos += 1
    return ghettos
