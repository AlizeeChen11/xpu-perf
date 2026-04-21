from pathlib import Path


def _load_version() -> str:
	version_file = Path(__file__).with_name("VERSION")
	try:
		return version_file.read_text(encoding="utf-8").strip()
	except OSError:
		return "0.0.0"


__version__ = _load_version()

__all__ = ["__version__"]
