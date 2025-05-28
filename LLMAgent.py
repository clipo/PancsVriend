import requests
import json
import config as cfg

def maybe_use_llm_agent(agent, r, c, grid):
    if not cfg.USE_LLM:
        return None

    context = extract_local_context(agent, r, c, grid)
    prompt = f"Given the agent is of type {agent.type_id} and the local context is:\n{context}\nWhere should it move (row,col)?"

    response = requests.post(
        cfg.OLLAMA_URL,
        headers={"Authorization": f"Bearer {cfg.OLLAMA_API_KEY}"},
        json={
            "model": cfg.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    try:
        text = response.json()['choices'][0]['message']['content']
        row, col = eval(text.strip())
        if 0 <= row < cfg.GRID_SIZE and 0 <= col < cfg.GRID_SIZE:
            return (row, col)
    except Exception:
        return None

def extract_local_context(agent, r, c, grid):
    context = []
    for dr in range(-1, 2):
        row = []
        for dc in range(-1, 2):
            r1, c1 = r + dr, c + dc
            if 0 <= r1 < cfg.GRID_SIZE and 0 <= c1 < cfg.GRID_SIZE:
                n = grid[r1][c1]
                if n is None:
                    row.append("E")
                elif n.type_id == agent.type_id:
                    row.append("S")
                else:
                    row.append("O")
            else:
                row.append("X")
        context.append(row)
    return json.dumps(context)
