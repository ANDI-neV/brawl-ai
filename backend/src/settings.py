import configparser
import os
from functools import lru_cache
from pathlib import Path


CONFIG_ENV_VAR = "BRAWL_AI_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.ini"


@lru_cache(maxsize=1)
def _load_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(get_config_path())
    return parser


def reset_settings_cache() -> None:
    _load_config.cache_clear()


def get_config_path() -> str:
    return os.getenv(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH))


def _env_name(section: str, option: str) -> str:
    return f"BRAWL_AI_{section}_{option}".upper()


def get_setting(
    section: str,
    option: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    env_value = os.getenv(_env_name(section, option))
    if env_value not in (None, ""):
        return env_value

    config = _load_config()
    if config.has_option(section, option):
        return config.get(section, option)

    if default is not None:
        return default

    if required:
        raise KeyError(
            f"Missing required configuration for [{section}] {option}. "
            f"Set {_env_name(section, option)} or update {get_config_path()}."
        )

    return ""


def get_int_setting(
    section: str,
    option: str,
    *,
    default: int,
) -> int:
    return int(get_setting(section, option, default=str(default)))


def get_db_config() -> dict:
    return {
        "host": get_setting("Credentials", "host", required=True),
        "port": get_int_setting("Credentials", "port", default=5432),
        "database": get_setting("Credentials", "database", required=True),
        "user": get_setting("Credentials", "username", required=True),
        "password": get_setting("Credentials", "password", required=True),
        "pool_min": get_int_setting("Credentials", "pool_min", default=1),
        "pool_max": get_int_setting("Credentials", "pool_max", default=10),
    }


def get_api_token() -> str:
    return get_setting("Credentials", "api", required=True)


def get_pi_config() -> dict:
    return {
        "pi_user": get_setting("Pi", "pi_user", default=""),
        "pi_host": get_setting("Pi", "pi_host", default=""),
        "pi_path": get_setting("Pi", "pi_path", default=""),
        "main_host": get_setting("Pi", "main_host", default=""),
        "public_ip": get_setting("Pi", "public_ip", default=""),
        "domain": get_setting("Pi", "domain", default=""),
    }


def get_cors_origins() -> list[str]:
    explicit_origins = os.getenv("BRAWL_AI_CORS_ORIGINS")
    if explicit_origins:
        return [origin.strip() for origin in explicit_origins.split(",") if origin.strip()]

    pi = get_pi_config()
    origins = {
        "http://localhost:3003",
        "http://127.0.0.1:3003",
    }

    if pi["pi_host"]:
        origins.add(f"http://{pi['pi_host']}:3003")
    if pi["main_host"]:
        origins.add(f"http://{pi['main_host']}:3000")
    if pi["public_ip"]:
        origins.add(f"http://{pi['public_ip']}:3003")
    if pi["domain"]:
        origins.update(
            {
                f"http://{pi['domain']}",
                f"https://{pi['domain']}",
                f"http://www.{pi['domain']}",
                f"https://www.{pi['domain']}",
            }
        )

    return sorted(origins)
