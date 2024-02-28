import pygame
import numpy as np
from jaxued.environments.maze.env import OBJECT_TO_INDEX, COLOR_TO_INDEX, COLORS, DIR_TO_VEC

TRI_COORDS = np.array([
    [0.12, 0.19],
    [0.87, 0.50],
    [0.12, 0.81],
])

class MazeVisualizer(object):
    def __init__(self, env, tile_size=32):
        self.screen = None
        self.clock = None
        self.env = env
        self.tile_size = tile_size
        self.atlas = _make_tile_atlas(tile_size)
        
    def get_frame(self, env_state, env_params, ignore_view=False):
        if hasattr(env_state, 'correct_answer') and hasattr(env_state, 'arbitrary_number'):
            img = np.zeros((5, 5), dtype=np.uint8)
            a = int(env_state.correct_answer   % 5)
            b = int(env_state.correct_answer // 5)
            img[a, b] = 255 * env_state.visible_answer + 128 * (1 - env_state.visible_answer)
            img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
            return img

        tile_size = self.tile_size
        nrows, ncols = self.env.max_height, self.env.max_width
        width_px, height_px = ncols * tile_size, nrows * tile_size

        f_vec = DIR_TO_VEC[env_state.agent_dir]
        r_vec = np.array([-f_vec[1], f_vec[0]])

        agent_view_size = self.env.agent_view_size
        fwd_bound1 = env_state.agent_pos
        fwd_bound2 = env_state.agent_pos + f_vec*(agent_view_size-1)
        side_bound1 = env_state.agent_pos - r_vec*(agent_view_size//2) 
        side_bound2 = env_state.agent_pos + r_vec*(agent_view_size//2)

        min_bound = np.min(np.stack([
            fwd_bound1, 
            fwd_bound2, 
            side_bound1, 
            side_bound2,
        ]), 0)

        min_y = min(max(min_bound[1], 0), env_state.wall_map.shape[1]-1) * tile_size
        min_x = min(max(min_bound[0], 0), env_state.wall_map.shape[0]-1) * tile_size
        max_y = min(max(min_bound[1]+agent_view_size, 0), env_state.wall_map.shape[1]) * tile_size
        max_x = min(max(min_bound[0]+agent_view_size, 0), env_state.wall_map.shape[0]) * tile_size
        
        # print(min_x, min_y, max_x, max_y)
        
        padding = agent_view_size - 1
        grid = env_state.maze_map[padding:-padding, padding:-padding, :]
        img = np.empty((height_px, width_px, 3), dtype=np.uint8)
        
        def get_bounds(x, y):
            ymin = y*tile_size
            ymax = (y+1)*tile_size
            xmin = x*tile_size
            xmax = (x+1)*tile_size
            return ymin, ymax, xmin, xmax

        for y in range(nrows):
            for x in range(ncols):
                obj = grid[y, x, :]
                obj_type = obj[0]
                
                if obj_type in [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty_diff']]:
                    tile = self.atlas[0]
                elif obj_type in [OBJECT_TO_INDEX['wall'], OBJECT_TO_INDEX['wall_diff']]:
                    tile = self.atlas[1]
                elif obj_type in [OBJECT_TO_INDEX['goal'], OBJECT_TO_INDEX['goal_diff']]:
                    tile = self.atlas[2]
                elif obj_type in [OBJECT_TO_INDEX['lava']]:
                    tile = self.atlas[9]
                elif obj_type == OBJECT_TO_INDEX['agent']:
                    agent_dir = obj[2]
                    tile = self.atlas[3 + agent_dir]
                
                ymin, ymax, xmin, xmax = get_bounds(x, y)
                img[ymin:ymax, xmin:xmax, :] = tile
                
        x, y = env_state.agent_pos
        img[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size, :] = self.atlas[3 + env_state.agent_dir]
        

        if hasattr(env_state, 'is_goal_hidden') and env_state.is_goal_hidden: # Show the goal as a slightly more dull green
            tile = np.tile([200, 255, 200], (tile_size, tile_size, 1))
            x, y = env_state.goal_pos
            ymin, ymax, xmin, xmax = get_bounds(x, y)
            img[ymin:ymax, xmin:xmax, :] = tile

        if hasattr(env_state, 'is_blind') and env_state.is_blind:
            T = 100
            blue = img[:, :, 2]
            idx = blue > 255 - T
            img[idx , 2] = 255
            img[~idx, 2] += T
        
        if not ignore_view:
            view_region = img[min_y:max_y, min_x:max_x, :]
            img[min_y:max_y, min_x:max_x, :] = (view_region + 0.3 * (255 - view_region)).clip(0, 255)
                
        return img
        
    def render(self, env_state, env_params):
        tile_size = self.tile_size
        nrows, ncols = self.env.max_height, self.env.max_width
        width_px, height_px = ncols * tile_size, nrows * tile_size
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((width_px, height_px))
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        surface = pygame.surfarray.make_surface(np.transpose(self.get_frame(env_state, env_params), (1, 0, 2)))                
        # surface = pygame.transform.flip(surface, False, True)

        assert self.screen is not None
        self.screen.blit(surface, (0, 0))
        pygame.event.pump()
        self.clock.tick(60)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
    
def _make_tile_atlas(tile_size):
    atlas = np.empty((10, tile_size, tile_size, 3), dtype=np.uint8)
    
    def add_border(tile):
        new_tile = fill_coords(tile, point_in_rect(0, 0.031, 0, 1), (100, 100, 100)) 
        return fill_coords(new_tile, point_in_rect(0, 1, 0, 0.031), (100, 100, 100)) 
    
    atlas[0] = add_border(np.tile([0, 0, 0], (tile_size, tile_size, 1))) # empty
    atlas[1] = np.tile([100, 100, 100], (tile_size, tile_size, 1)) # wall
    atlas[2] = np.tile([0, 255, 0], (tile_size, tile_size, 1)) # goal
    
    # Handle player
    agent_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    agent_tile = fill_coords(agent_tile, point_in_triangle(*TRI_COORDS), [255, 0, 0])
    
    atlas[3] = add_border(agent_tile) # right
    atlas[4] = add_border(np.rot90(agent_tile, k=3)) # down
    atlas[5] = add_border(np.rot90(agent_tile, k=2)) # left
    atlas[6] = add_border(np.rot90(agent_tile, k=1)) # up
    

    atlas[9] = np.tile([255, 69, 0], (tile_size, tile_size, 1)) # lava
    return atlas

def fill_coords(img, fn, color):
    new_img = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                new_img[y, x] = color
    return new_img

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn

def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn