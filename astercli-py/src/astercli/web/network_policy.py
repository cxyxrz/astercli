"""网络访问策略（对齐 Java web.NetworkPolicy）。

三道围栏：
- scheme 白名单：仅允许 http / https
- 主机黑名单：屏蔽 localhost、0.0.0.0 以及 DNS 解析后的环回 / 未指定 /
  链路本地 / 内网地址（防 SSRF）
- 简易窗口限流：每 60 秒最多 30 次请求

与 Java 的差异：site-local 判定用 ``ipaddress.is_private``（覆盖 10/8、
172.16/12、192.168/16 及更多保留网段，防 SSRF 语义只严不松）。
当前是基础围栏，DNS rebinding 防护等严苛企业环境能力暂不覆盖。
"""

from __future__ import annotations

import ipaddress
import socket
import time
from typing import Optional
from urllib.parse import urlsplit

ALLOWED_SCHEMES = ("http", "https")
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_MAX_PER_WINDOW = 30


class NetworkPolicy:
    def __init__(
        self,
        *,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        max_per_window: int = DEFAULT_MAX_PER_WINDOW,
    ) -> None:
        self._window_seconds = window_seconds
        self._max_per_window = max_per_window
        self._window_start: float = 0.0
        self._counter = 0

    # ---------- URL 校验 ----------

    def check_url(self, url: str) -> Optional[str]:
        """校验 URL 是否允许访问。返回 None 表示通过，否则返回拒绝原因。"""
        if not url or not url.strip():
            return "URL 不能为空"
        try:
            parts = urlsplit(url.strip())
        except ValueError as e:
            return f"URL 格式非法: {e}"

        scheme = (parts.scheme or "").lower()
        if not scheme:
            return "URL 缺少 scheme（需 http 或 https）"
        if scheme not in ALLOWED_SCHEMES:
            return f"禁止的 scheme: {scheme}（仅允许 http、https）"

        host = parts.hostname
        if not host:
            return "URL 缺少 host"

        return self._check_host(host)

    def _check_host(self, host: str) -> Optional[str]:
        lower = host.lower()
        # 字面量黑名单（即便 DNS 解析失败也直接拒绝）
        if lower == "localhost" or lower.endswith(".localhost"):
            return "禁止访问 localhost"
        if lower == "0.0.0.0":
            return "禁止访问 0.0.0.0"

        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return f"无法解析主机: {host}"

        seen: set = set()
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
            except ValueError:
                continue
            if ip in seen:
                continue
            seen.add(ip)
            if ip.is_loopback:
                return f"禁止访问环回地址（{ip}）"
            if ip.is_unspecified:
                return f"禁止访问未指定地址（{ip}）"
            if ip.is_link_local:
                return f"禁止访问链路本地地址（{ip}）"
            if ip.is_private:
                return f"禁止访问内网地址（{ip}）"
        return None

    # ---------- 限流 ----------

    def acquire(self) -> Optional[str]:
        """申请一次请求配额。返回 None 表示通过，否则返回限流原因。"""
        now = time.monotonic()
        if self._window_start <= 0 or now - self._window_start >= self._window_seconds:
            self._window_start = now
            self._counter = 1
            return None
        self._counter += 1
        if self._counter > self._max_per_window:
            reset_in = max(int((self._window_seconds - (now - self._window_start))), 1)
            return (
                f"请求过于频繁（{self._window_seconds} 秒内已达 {self._max_per_window} 次上限），"
                f"约 {reset_in} 秒后重试"
            )
        return None
