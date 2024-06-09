import yaml

#===============================================================================
# Load YAML configuration file
#===============================================================================
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def update_default_configs(default_config, yaml_config):
    for key, value in yaml_config.items():
        if key in default_config:
            # Check the type of the default value
            default_value_type = type(default_config[key])
            
            # Handle string values that need to be converted to floats or ints
            if isinstance(value, str) and default_value_type in [float, int]:
                try:
                    if default_value_type is float:
                        value = float(value)
                    elif default_value_type is int:
                        value = int(value)
                except ValueError:
                    # If conversion fails, raise an error
                    raise ValueError(f"Cannot convert value '{value}' to type {default_value_type} for key '{key}'")
            elif not isinstance(value, default_value_type):
                # If the types are incompatible, raise an error
                raise TypeError(f"Expected value of type {default_value_type} for key '{key}', but got {type(value)}")
            
            # Update the default configuration with the new value
            default_config[key] = value
        else:
            # If the key is not in the default configuration, raise an error
            raise ValueError(f"Invalid configuration key: {key}")
    
    return default_config

#===============================================================================
# Colorful print
#===============================================================================
def print_color(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'end': '\033[0m'
    }
    print(f"{colors[color]}{text}{colors['end']}")