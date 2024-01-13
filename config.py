from pydantic import BaseModel
import yaml
from pydantic import ValidationError


class FilesConfig(BaseModel):
    output_model_h5: str


class StorageConfig(BaseModel):
    bucket_name: str
    files: FilesConfig


class DataConfig(BaseModel):
    refresh: bool


class ModelConfig(BaseModel):
    refresh: bool


class AppConfig(BaseModel):
    storage: StorageConfig
    data: DataConfig
    model: ModelConfig


# Load YAML content
with open('config.yaml', 'r') as file:
    yaml_data = yaml.safe_load(file)
# Parse YAML data using pydantic
try:
    app_config = AppConfig.model_validate(yaml_data)
    print(app_config.storage)

except ValidationError as e:
    print(f"Validation error: {e}")
