import yaml

file_path = "/home/ashish/ros2/avinya_ws/src/ai_assist/config.yaml"

# Read and parse the YAML file
with open(file_path, 'r') as file:
    config_data = yaml.safe_load(file)

# Initialize an empty dictionary
pre_dock_position = {}

# Process the pre_dock_position data from YAML
for item in config_data['pre_dock_position']:
    for key, value in item.items():
        pre_dock_position[key] = {'x': value[0], 'y': value[1], 'yaw': value[2]}

# Print the extracted dictionary
print(pre_dock_position)
