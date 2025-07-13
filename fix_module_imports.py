import os

dirs_needing_init = [
    '/home/user/tiktok_production/utils',
    '/home/user/tiktok_production/configs',
    '/home/user/tiktok_production/analyzers',
    '/home/user/tiktok_production/api'
]

for dir_path in dirs_needing_init:
    init_file = os.path.join(dir_path, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Auto-generated __init__.py\n')
        print(f"Created: {init_file}")
    else:
        print(f"Already exists: {init_file}")