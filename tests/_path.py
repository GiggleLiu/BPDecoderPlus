import os
import sys


def add_project_root_to_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_root = os.path.join(project_root, "src")
    for path in (src_root, project_root):
        if path not in sys.path:
            sys.path.insert(0, path)
