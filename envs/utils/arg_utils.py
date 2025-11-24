import argparse

def normalize_args(args):
    """Convert argument names from hyphenated to underscored format"""
    arg_dict = vars(args)
    normalized = {}
    
    for key, value in arg_dict.items():
        # Replace hyphens with underscores
        normalized_key = key.replace('-', '_')
        normalized[normalized_key] = value
    
    # Create new Namespace with normalized keys
    return argparse.Namespace(**normalized)