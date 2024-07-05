from pathlib import Path

import structlog
from config import ConfigurationSet, config_from_env, config_from_toml
from dotenv import find_dotenv, load_dotenv

from .models import init_embed_model, init_model

log = structlog.get_logger()


def init():
    load_dotenv(find_dotenv(".env"))
    conf_path = Path(__file__).parent / "config.toml"
    return ConfigurationSet(
        config_from_env("GENRAITOR", separator="__", lowercase_keys=True),
        config_from_toml(conf_path, read_from_file=True),
    )


env = init()

# llm_model = init_model(env)
# embed_model = init_embed_model(env)
