import numpy as np

def load_vertex_colors(obj_path):
    v_colors = []
    for line in open(obj_path, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v_colors.append(values[4:7])
        else:
            continue
    return np.asarray(v_colors, dtype=np.float32)
