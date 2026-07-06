#!/bin/sh

git clone https://github.com/Fyusion/LLFF.git
pip install scikit-image
# --- AUTO-PATCHER FOR LLFF COLMAP BUGS ---
python -c "
fpath = 'LLFF/llff/poses/pose_utils.py'
with open(fpath, 'r') as f: code = f.read()

# Fix 1: Dropped Camera Out-Of-Bounds Crash
old_cams = 'if len(cams) < ind - 1:\n                print(\'ERROR: the correct camera poses for current points cannot be accessed\')\n                return'
new_cams = 'if ind - 1 >= len(cams):\n                continue'
code = code.replace(old_cams, new_cams)

# Fix 2: Zero-size array minimum crash (Global Depth)
old_vz = 'valid_z = zvals[vis_arr==1]\n    print( \'Depth stats\', valid_z.min(), valid_z.max(), valid_z.mean() )'
new_vz = '''valid_z = zvals[vis_arr==1]
    if len(valid_z) == 0:
        valid_z = zvals.flatten()
        print('Depth stats (fallback)', valid_z.min(), valid_z.max(), valid_z.mean())
    else:
        print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())'''
code = code.replace(old_vz, new_vz)

# Fix 3: Empty Depth Array Percentile Crash (Local Depth)
old_depth = 'close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)'
new_depth = '''if len(zs) == 0:\n            close_depth, inf_depth = valid_z.min(), valid_z.max()\n        else:\n            close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)'''
code = code.replace(old_depth, new_depth)

with open(fpath, 'w') as f: f.write(code)
"
# -----------------------------------------
