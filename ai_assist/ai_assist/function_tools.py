import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# function_tools.py

first_tools = [
    {
        "type": "function",
        "function": {
            "name": "mov_nav",
            "description": "Move the robot with optional predefined location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "linear_x": {"type": "number", "description": "Movement in X-axis."},
                    "linear_y": {"type": "number", "description": "Movement in Y-axis."},
                    "linear_z": {"type": "number", "description": "Rotation in radians."},
                    "checkpoint": {"type": "string", "description": "Optional predefined location."}
                },
                "required": ["linear_x", "linear_y", "linear_z"],
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
