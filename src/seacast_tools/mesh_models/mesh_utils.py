import os
import shutil


def move_graphfiles_to_active_directory(source_directory, target_directory):
    """
    Move all graph files from the source directory to the target directory.
    
    Args:
        source_directory (str): The directory containing the graph files to be moved.
        target_directory (str): The directory where the graph files will be moved.
    """

    os.makedirs(target_directory, exist_ok=True)

    for filename in os.listdir(target_directory):
        if filename.endswith(".pt"):
            os.remove(os.path.join(target_directory, filename))

    for filename in os.listdir(source_directory):
        if filename.endswith(".pt"):
            src = os.path.join(source_directory, filename)
            dst = os.path.join(target_directory, filename)
            shutil.copy2(src, dst)
