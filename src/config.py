import os
from dotenv import load_dotenv
from pathlib import Path

class Config:
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self._load_environment()

    def _load_environment(self):
        """Load the appropriate .env file based on environment"""
        env_file = Path(f'.env.{self.environment}')
        if env_file.exists():
            load_dotenv(env_file)
        else:
            print(f"Warning: {env_file} not found!")

    @property
    def db_config(self):
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": os.getenv("DB_PORT", "5432"),
            "schema": os.getenv("DB_SCHEMA", "app_schema")
        }

    @property
    def is_development(self):
        return self.environment == 'development'

    @property
    def is_test(self):
        return self.environment == 'test'

    @property
    def is_production(self):
        return self.environment == 'production'