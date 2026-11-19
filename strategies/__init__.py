"""待选策略：信号到持仓的规则与组合定义。回测在 backtest，因子在 research。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
