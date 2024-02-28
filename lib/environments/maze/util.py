from jaxued.environments.maze.env import OBJECT_TO_INDEX, COLOR_TO_INDEX, COLORS, DIR_TO_VEC
import jax.numpy as jnp

def pad_maze_map(maze_map, padding):
    empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
    wall = jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0], dtype=jnp.uint8)
    goal = jnp.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0], dtype=jnp.uint8)

    maze_map_padded = jnp.tile(wall.reshape((1, 1, *empty.shape)), (maze_map.shape[0]+2*padding, maze_map.shape[1]+2*padding, 1))
    maze_map_padded = maze_map_padded.at[padding:-padding,padding:-padding,:].set(maze_map)

    # Add surrounding walls
    wall_start = padding-1 # start index for walls
    wall_end_y = maze_map_padded.shape[0] - wall_start - 1
    wall_end_x = maze_map_padded.shape[1] - wall_start - 1
    maze_map_padded = maze_map_padded.at[wall_start,wall_start:wall_end_x+1,:].set(wall) # top
    maze_map_padded = maze_map_padded.at[wall_end_y,wall_start:wall_end_x+1,:].set(wall) # bottom
    maze_map_padded = maze_map_padded.at[wall_start:wall_end_y+1,wall_start,:].set(wall) # left
    maze_map_padded = maze_map_padded.at[wall_start:wall_end_y+1,wall_end_x,:].set(wall) # right

    return maze_map_padded

