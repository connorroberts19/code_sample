import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, os.path.abspath(parent_dir))