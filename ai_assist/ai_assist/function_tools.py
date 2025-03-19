import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# function_tools.py

first_tools = [
    {
        "type": "function",
        "function": {
            "name": "mov_cmd",
            "description": "Move the robot a specified distance and/or rotate to a given angle in radians.",
            "parameters": {
                "type": "object",
                "properties": {
                    "linear_x": {"type": "number", "description": "Distance to move."},
                    "angular_z": {"type": "number", "description": "Angle to turn in radians."},
                },
                "required": ["linear_x", "angular_z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture",
            "description": "Capture an image using the camera.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

second_tools = [
    {
        "type": "function",
        "function": {
            "name": "nav_goal",
            "description": "Move the robot to a  location given by the user, the location should not be none .",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": " location name."},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "docking",
            "description": "Dock the robot at a predefined docking station.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Immediately stop all robot actions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]