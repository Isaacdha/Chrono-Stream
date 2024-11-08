import os

def generate_tree_structure(output_file='directory_structure.txt'):
    # Start from the current working directory
    root_dir = os.getcwd()
    
    with open(output_file, 'w') as file:
        create_tree(root_dir, file)
    
    print(f"Directory structure written to {output_file}")

def create_tree(folder_path, file, prefix=''):
    # List all items in the folder
    items = sorted(os.listdir(folder_path))
    # Separate folders and files
    folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    files = [item for item in items if os.path.isfile(os.path.join(folder_path, item))]

    # Iterate over folders and files
    for index, folder in enumerate(folders):
        connector = "├── " if index < len(folders) - 1 or files else "└── "
        file.write(f"{prefix}{connector}{folder}/\n")
        # Recurse into subfolders with updated prefix
        new_prefix = prefix + ("│   " if index < len(folders) - 1 or files else "    ")
        create_tree(os.path.join(folder_path, folder), file, new_prefix)

    for index, filename in enumerate(files):
        connector = "└── " if index == len(files) - 1 else "├── "
        file.write(f"{prefix}{connector}{filename}\n")

# Run the function to generate the structure
generate_tree_structure()