import yaml
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str


@dataclass
class OpenAIConfig:
    api_key: str
    api_base: str


@dataclass
class AppConfig:
    neo4j: Neo4jConfig
    openai: OpenAIConfig


@dataclass
class UIClientConfig:
    host: str
    port: int


@dataclass
class ModelServerConfig:
    host: str
    port: int


class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        self.neo4j = Neo4jConfig(
            uri=raw_config["neo4j"]["uri"],
            username=raw_config["neo4j"]["username"],
            password=raw_config["neo4j"]["password"]
        )

        self.openai = OpenAIConfig(
            api_key=raw_config["openai"]["api_key"],
            api_base=raw_config["openai"]["api_base"]
        )

        self.ui_client = UIClientConfig(
            host=raw_config["ui_client"]['host'],
            port=raw_config['ui_client']['port']
        )

        self.model_server = ModelServerConfig(
            host=raw_config['model_server']['host'],
            port=raw_config['model_server']['port']
        )
