import argparse
import json
import os
import random
from datetime import datetime

import config as cfg
import requests
from context_scenarios import CONTEXT_SCENARIOS


def build_grid(grid_size, num_type_a, num_type_b, center_pos, center_type_id, seed=None):
    if seed is not None:
        random.seed(seed)

    grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    r_center, c_center = center_pos
    grid[r_center][c_center] = _Agent(center_type_id)

    remaining = []
    remaining.extend([_Agent(0) for _ in range(num_type_a)])
    remaining.extend([_Agent(1) for _ in range(num_type_b)])

    # Remove one agent of the center type if possible (already placed)
    for i, agent in enumerate(remaining):
        if agent.type_id == center_type_id:
            remaining.pop(i)
            break

    empty_positions = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r][c] is None]
    random.shuffle(empty_positions)

    for agent, pos in zip(remaining, empty_positions):
        r, c = pos
        grid[r][c] = agent

    return grid


def get_context_grid(r, c, grid):
    context = []
    for dr in [-1, 0, 1]:
        row = []
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                neighbor = grid[nr][nc]
                if neighbor is None:
                    row.append("E")
                elif neighbor.type_id == grid[r][c].type_id:
                    row.append("S")
                else:
                    row.append("O")
            else:
                row.append("#")
        context.append(row)

    context_with_position = []
    for i, row in enumerate(context):
        new_row = []
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                new_row.append("X")
            else:
                new_row.append(cell)
        context_with_position.append(new_row)

    return "\n".join([" ".join(row) for row in context_with_position])


def build_prompt(scenario, center_type_id, context_str):
    context_info = CONTEXT_SCENARIOS[scenario]
    agent_type = context_info['type_a'] if center_type_id == 0 else context_info['type_b']
    opposite_type = context_info['type_b'] if center_type_id == 0 else context_info['type_a']
    return context_info['prompt_template'].format(
        agent_type=agent_type,
        opposite_type=opposite_type,
        context=context_str
    )


def request_llm_response(prompt, llm_model, llm_url, llm_api_key, temperature, max_tokens, timeout):
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": int(timeout * 1000)
    }
    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(llm_url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


class _Agent:
    def __init__(self, type_id):
        self.type_id = type_id


def main():
    parser = argparse.ArgumentParser(description="Debug helper: build and send an LLM prompt for a scenario")
    parser.add_argument('--scenario', type=str, required=True, choices=list(CONTEXT_SCENARIOS.keys()))
    parser.add_argument('--row', type=int, default=None)
    parser.add_argument('--col', type=int, default=None)
    parser.add_argument('--type-id', type=int, choices=[0, 1], default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--llm-url', type=str, default=None)
    parser.add_argument('--llm-api-key', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('--dry-run', action='store_true', help='Only print the prompt, do not call the LLM')
    args = parser.parse_args()

    llm_model = args.llm_model or cfg.OLLAMA_MODEL
    llm_url = args.llm_url or cfg.OLLAMA_URL
    llm_api_key = args.llm_api_key or cfg.OLLAMA_API_KEY

    row = args.row if args.row is not None else random.randrange(cfg.GRID_SIZE)
    col = args.col if args.col is not None else random.randrange(cfg.GRID_SIZE)
    type_id = args.type_id if args.type_id is not None else random.choice([0, 1])

    if not (0 <= row < cfg.GRID_SIZE and 0 <= col < cfg.GRID_SIZE):
        raise ValueError(f"row/col must be within 0..{cfg.GRID_SIZE - 1}")

    grid = build_grid(
        grid_size=cfg.GRID_SIZE,
        num_type_a=cfg.NUM_TYPE_A,
        num_type_b=cfg.NUM_TYPE_B,
        center_pos=(row, col),
        center_type_id=type_id,
        seed=args.seed
    )

    context_str = get_context_grid(row, col, grid)
    prompt = build_prompt(args.scenario, type_id, context_str)

    print("---PROMPT---")
    # print(f"[debug] selected center: row={row}, col={col}, type_id={type_id}")
    print(prompt)
    print(context_str)

    if args.dry_run:
        return

    print("---RESPONSE---")
    for i in range(10):  # Make multiple requests to test variability
        # prompt = "MOVE or STAY?"
        response_text = request_llm_response(
            prompt=prompt,
            llm_model=llm_model,
            llm_url=llm_url,
            llm_api_key=llm_api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout
        )
        print(response_text)


if __name__ == "__main__":
    main()
