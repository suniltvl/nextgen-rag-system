from pathlib import Path


class Helper:
    def __init__(self):
        pass
    
    def validate_url(self, url: str) -> bool:
        # TODO: Implement URL validation logic
        return True

    def extract_domain(self, url: str) -> str:
        # TODO: Implement domain extraction logic
        return url

    def get_project_root(self) -> Path:
        project_root = Path.cwd().resolve()
        while not (project_root / "pyproject.toml").exists() and project_root != project_root.parent:
            project_root = project_root.parent
        return project_root

    def is_dir_in_project(self, dir_name: str) -> bool:
        """
        Check if a directory exists within the project root.
        
        Args:
            dir_name: Name of the directory to check
            
        Returns:
            True if the directory exists, False otherwise
        """
        dir_path = self.get_project_root() / dir_name
        return dir_path.exists() and dir_path.is_dir()

    def get_dir_in_project(self, dir_name: str) -> Path:
        """
        Get the path to a directory within the project root.
        
        Args:
            dir_name: Name of the directory to find
            
        Returns:
            Path to the directory
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
            NotADirectoryError: If the path exists but is not a directory
        """
        dir_path = self.get_project_root() / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory {dir_name} not found in project root")
        elif not dir_path.is_dir():
            raise NotADirectoryError(f"{dir_name} is not a directory")
        return dir_path

    def create_dir(self, dir_name: str) -> Path:
        """
        Create a directory within the project root.
        
        Args:
            dir_name: Name of the directory to create
            
        Returns:
            Path to the created directory
        """
        dir_path = self.get_project_root() / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


helper = Helper()
