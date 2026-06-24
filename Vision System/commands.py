from dataclasses import dataclass
from typing import Callable


@dataclass
class Command:
    name: str
    description: str
    handler: Callable



def cmd_save(args, state):
    print("Saving config...")


def cmd_show_config(args, state):
    print(state)


def cmd_help(args, state):
    for command in COMMANDS.values():
        print(f"{command.name:<15} {command.description}")



COMMANDS = {
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