from pathlib import Path
import yaml
from src.models import RAGConfig
from src.utils.helper import helper

# def load_config(config_path: str) -> RAGConfig:
#     with open(config_path, "r", encoding="utf-8") as f:
#         data = yaml.safe_load(f)

#     # return RAGConfig(**data) 
#     return RAGConfig.model_validate(data)


project_root = helper.get_project_root()
print(f"Project root: {project_root}")

def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()

    for key, value in override.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)

        else:
            result[key] = value

    return result


def load_config(path: str) -> RAGConfig:

    path = Path(project_root, path)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if "extends" in config:

        parent_path = Path(path.parent, config["extends"])
        # print(f"Parent path: {parent_path}")
        with open(parent_path, "r") as f:
            parent = yaml.safe_load(f)
            # print(f"Parent config: {parent}")

        config.pop("extends")

        config = deep_merge(parent, config)

    # print(f"Final config: {config}")
    return RAGConfig.model_validate(config)