import warnings
from .entropy_analyzer import EntropyAnalyzer
from .version import Version

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main",
)

__version__ = Version.ENTROPY_ANALYZER_CURRENT_VERSION
__all__ = ["EntropyAnalyzer"]
