

try:
    from pathlib import Path
    import sys

    def get_project_root():
        project_root = Path.cwd().resolve()
        while not (project_root / "pyproject.toml").exists() and project_root != project_root.parent:
            project_root = project_root.parent
        return project_root

    def setup_notebook():
        project_root = get_project_root()
        print(f"✓ Project root: {project_root}")
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

    print("✓ Notebook environment setup completed successfully")
except Exception as e:
    print(f"✗ Error setting up notebook environment: {str(e)}")
    raise