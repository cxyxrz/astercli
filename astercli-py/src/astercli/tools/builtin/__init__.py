"""内置工具（对齐 Java ToolRegistry 内置工具的参数与输出格式）。"""

from .files import register_file_tools
from .shell import register_shell_tools

__all__ = ["register_file_tools", "register_shell_tools"]
