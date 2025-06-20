

from can_llm.model_server import ModelServer
from can_llm.utils import Config

if __name__ == '__main__':
    config_s = Config('config/config.yaml')
    server = ModelServer(config=config_s)
    server.start_server()
