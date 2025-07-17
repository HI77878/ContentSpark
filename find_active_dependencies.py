#!/usr/bin/env python3
"""
Find active dependencies - which files are actually imported by the production system
"""

import ast
import os
import sys
from pathlib import Path
from typing import Set, List

def extract_imports(file_path: str) -> Set[str]:
    """Extract all imports from a Python file"""
    imports = set()
    
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    for alias in node.names:
                        imports.add(f"{node.module}.{alias.name}")
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return imports

def find_dependencies(start_file: str, project_root: str) -> Set[str]:
    """Recursively find all dependencies starting from a file"""
    
    visited = set()
    to_visit = {start_file}
    all_files = set()
    
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
            
        visited.add(current)
        
        if os.path.exists(current):
            all_files.add(current)
            imports = extract_imports(current)
            
            # Convert imports to potential file paths
            for imp in imports:
                # Handle local imports
                if imp.startswith('analyzers.') or imp.startswith('utils.') or imp.startswith('configs.'):
                    module_path = imp.replace('.', '/') + '.py'
                    full_path = os.path.join(project_root, module_path)
                    if os.path.exists(full_path):
                        to_visit.add(full_path)
                
                # Handle relative imports in the same directory
                elif not imp.startswith('_'):
                    # Check in project root
                    root_path = os.path.join(project_root, imp + '.py')
                    if os.path.exists(root_path):
                        to_visit.add(root_path)
                    
                    # Check in same directory as current file
                    dir_path = os.path.dirname(current)
                    local_path = os.path.join(dir_path, imp + '.py')
                    if os.path.exists(local_path):
                        to_visit.add(local_path)
    
    return all_files

def main():
    project_root = '/home/user/tiktok_production'
    
    # Start from the main production API
    start_files = [
        os.path.join(project_root, 'api/stable_production_api.py'),
        os.path.join(project_root, 'ml_analyzer_registry_complete.py'),
    ]
    
    all_active_files = set()
    
    for start_file in start_files:
        print(f"\nAnalyzing dependencies for: {start_file}")
        deps = find_dependencies(start_file, project_root)
        all_active_files.update(deps)
    
    # Get all Python files in the project
    all_python_files = set()
    for root, dirs, files in os.walk(project_root):
        # Skip virtual environments and cache
        if 'venv' in root or '__pycache__' in root or '.cache' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                all_python_files.add(os.path.join(root, file))
    
    # Find inactive files
    inactive_files = all_python_files - all_active_files
    
    # Categorize inactive files
    inactive_by_category = {
        'analyzers': [],
        'utils': [],
        'api': [],
        'tests': [],
        'aurora_cap': [],
        'other': []
    }
    
    for file in sorted(inactive_files):
        rel_path = os.path.relpath(file, project_root)
        
        if 'test' in rel_path.lower() or rel_path.startswith('test_'):
            inactive_by_category['tests'].append(rel_path)
        elif rel_path.startswith('analyzers/'):
            inactive_by_category['analyzers'].append(rel_path)
        elif rel_path.startswith('utils/'):
            inactive_by_category['utils'].append(rel_path)
        elif rel_path.startswith('api/'):
            inactive_by_category['api'].append(rel_path)
        elif rel_path.startswith('aurora_cap/'):
            inactive_by_category['aurora_cap'].append(rel_path)
        else:
            inactive_by_category['other'].append(rel_path)
    
    # Print results
    print("\n" + "="*80)
    print(f"ACTIVE FILES: {len(all_active_files)} files")
    print("="*80)
    
    print("\nActive analyzers:")
    for file in sorted(all_active_files):
        rel_path = os.path.relpath(file, project_root)
        if rel_path.startswith('analyzers/'):
            print(f"  - {rel_path}")
    
    print("\n" + "="*80)
    print(f"INACTIVE FILES: {len(inactive_files)} files (potentially removable)")
    print("="*80)
    
    for category, files in inactive_by_category.items():
        if files:
            print(f"\n{category.upper()} ({len(files)} files):")
            for file in sorted(files)[:20]:  # Show first 20
                print(f"  - {file}")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more")
    
    # Save detailed report
    with open(os.path.join(project_root, 'active_dependencies_report.txt'), 'w') as f:
        f.write("ACTIVE FILES (Used by production system):\n")
        f.write("="*80 + "\n")
        for file in sorted(all_active_files):
            f.write(f"{os.path.relpath(file, project_root)}\n")
        
        f.write("\n\nINACTIVE FILES (Potentially removable):\n")
        f.write("="*80 + "\n")
        for category, files in inactive_by_category.items():
            if files:
                f.write(f"\n{category.upper()}:\n")
                for file in sorted(files):
                    f.write(f"{file}\n")
    
    print(f"\nDetailed report saved to: active_dependencies_report.txt")

if __name__ == "__main__":
    main()