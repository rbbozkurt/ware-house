import itertools

from gym.envs.registration import register
from ware_house.envs import WarehouseEnv


_sizes = {
    "tiny": (4, 4),
    "small": (6, 6),
    "medium": (8, 8),
    "large": (10, 10),
}
_agents = {"1": 1 , "2" : 2 , "3" : 4, "4" : 8}
_shelves = {"1": 2 , "2" : 4 , "3" : 8, "4" : 16}
_variations = {"1": 1, "2": 2, "3": 3, "4":4}

_settings = zip(_variations.keys(),_sizes.keys(), _agents.keys(), _shelves.keys())

for _variation, _size, _agent_number, _shelf_number in _settings:
    # normal tasks
    register(
        id=f"rware-v{_variations[_variation]}",
        entry_point='ware_house.envs:WarehouseEnv',
        kwargs={
            "width": _sizes[_size][0],
            "height": _sizes[_size][1],
            "n_agents": _agents[_agent_number],
            "n_shelves": _shelves[_shelf_number]
        }
    )