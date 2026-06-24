from dataclasses import dataclass
from typing import Callable
from processing import save_config, cfg

@dataclass
class Command:
    name: str
    description: str
    handler: Callable


def cmd_check(args, state):
    print("Works")

def cmd_save(args, state):
    # sync trackbar values into cfg before saving
    from processing import cfg
    params = state.get("circle_params")
    if params:
        cfg.circles.dp         = params["dp"]
        cfg.circles.min_dist   = params["min_dist"]
        cfg.circles.param1     = params["param1"]
        cfg.circles.param2     = params["param2"]
        cfg.circles.min_radius = params["min_radius"]
        cfg.circles.max_radius = params["max_radius"]
    save_config(cfg)
    print("Config saved.")

def cmd_show_config(args, state):
    print(state)


def cmd_help(args, state):
    for command in COMMANDS.values():
        print(f"{command.name:<15} {command.description}")

def cmd_points_array(args, state):
    points_array = state.get("final_points_array", [])
    print("Final Points Array:")
    for point in points_array:
        print(point)
    print(f"Total points: {len(points_array)}")


COMMANDS = {
    "check": Command(
        name="check",
        description="Check if the command system is working",
        handler=cmd_check
    ),
    "save": Command(
        name="save",
        description="Save configuration to yaml",
        handler=cmd_save
    ),

    "show_config": Command(
        name="show_config",
        description="Display current configuration",
        handler=cmd_show_config
    ),

    "help": Command(
        name="help",
        description="Show available commands",
        handler=cmd_help
    ),

    "points_array": Command(
        name="points_array",
        description="Display the final points array",
        handler=cmd_points_array
    )
}

def execute_command(command_line, state):
    tokens = command_line.split()

    if not tokens:
        return

    command_name = tokens[0]
    args = tokens[1:]

    command = COMMANDS.get(command_name)

    if command is None:
        print(f"Unknown command: {command_name}")
        return

    command.handler(args, state)