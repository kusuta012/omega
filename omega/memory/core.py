import logging
from pathlib import Path
from omega.environment.conf_loader import omega_settings

logger = logging.getLogger("CoreFiles")

MEMORY_SEED = """# Omega Memory
No memories yet. This file will be populated as conversations accumulate.
"""

USER_SEED = """# User Profile
No profile information yet. This file will be populated as Omega learns about you.
"""

def ensure_memory_dir():
    path = Path(omega_settings.memory_dir)
    path.mkdir(parents=True, exist_ok=True)

    memory_path = path / "MEMORY.md"
    user_path = path / "USER.md"

    if not memory_path.exists():
        memory_path.write_text(MEMORY_SEED)
        logger.info(f"Created seed MEMORY.md at {memory_path}")

    if not user_path.exists():
        user_path.write_text(USER_SEED)
        logger.info(f"Created seed USER.md at {user_path}")

    return path

def read_memory_md() -> str:
    path = Path(omega_settings.memory_dir) / "MEMORY.md"
    if not path.exists():
        ensure_memory_dir()
    return path.read_text()

def read_user_md() -> str:
    path = Path(omega_settings.memory_dir) / "USER.md"
    if not path.exists():
        ensure_memory_dir()
    return path.read_text()