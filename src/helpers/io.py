import os
import time

def generate_output_path():
    project_root = os.getcwd()
    output_folder = os.path.join(project_root, 'images', f'run-{int(round(time.time() * 1000))}')
    os.makedirs(output_folder, exist_ok=True)
    return output_folder