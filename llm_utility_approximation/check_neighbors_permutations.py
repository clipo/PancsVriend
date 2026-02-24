
from itertools import product
import csv


SCHELLING_GRID_SIZE = 10
NON_WALL_CONTEXT_ELEMENTS = ["S", "O", "E"]
NEIGHBOR_OFFSETS = [
	(-1, -1),
	(-1, 0),
	(-1, 1),
	(0, -1),
	(0, 1),
	(1, -1),
	(1, 0),
	(1, 1),
]

def _wall_mask_for_position(row: int, col: int, grid_size: int) -> tuple[bool, ...]:
	mask: list[bool] = []
	for dr, dc in NEIGHBOR_OFFSETS:
		nr = row + dr
		nc = col + dc
		mask.append(not (0 <= nr < grid_size and 0 <= nc < grid_size))
	return tuple(mask)
def _all_valid_wall_masks(grid_size: int) -> list[tuple[bool, ...]]:
	unique_masks: set[tuple[bool, ...]] = set()
	for row in range(grid_size):
		for col in range(grid_size):
			unique_masks.add(_wall_mask_for_position(row, col, grid_size))
	return sorted(unique_masks)
def generate_all_valid_schelling_neighbors(grid_size: int = SCHELLING_GRID_SIZE) -> list[list[str]]:
	arrangements: list[list[str]] = []
	for wall_mask in _all_valid_wall_masks(grid_size):
		open_indices = [idx for idx, is_wall in enumerate(wall_mask) if not is_wall]
		for open_values in product(NON_WALL_CONTEXT_ELEMENTS, repeat=len(open_indices)):
			neighbors = ["#" if wall_mask[idx] else "" for idx in range(8)]
			for slot_idx, symbol in zip(open_indices, open_values):
				neighbors[slot_idx] = symbol
			arrangements.append(neighbors)
	return arrangements

def save_neighbors_to_csv(filename: str = "neighbors.csv", grid_size: int = SCHELLING_GRID_SIZE) -> None:
	neighbors = generate_all_valid_schelling_neighbors(grid_size)
	with open(filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow([f"neighbor_{i}" for i in range(8)])
		writer.writerows(neighbors)

if __name__ == "__main__":
	save_neighbors_to_csv()

