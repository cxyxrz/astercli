"""NetworkPolicy 测试：URL 校验围栏 + 限流（零网络，DNS 全 mock）。"""

from __future__ import annotations

import socket

import pytest

from astercli.web.network_policy import NetworkPolicy


def fake_resolver(ips):
    """构造 getaddrinfo 假实现：把任意 host 解析为指定 IP 列表。"""

    def resolve(host, *args, **kwargs):
        infos = []
        for ip in ips:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            infos.append((family, socket.SOCK_STREAM, 6, "", (ip, 0)))
        return infos

    return resolve


@pytest.fixture
def policy():
    return NetworkPolicy()


class TestCheckUrl:
    def test_空_url_拒绝(self, policy):
        assert policy.check_url("") == "URL 不能为空"

    def test_缺_scheme_拒绝(self, policy):
        assert "缺少 scheme" in policy.check_url("example.com/x")

    def test_非法_scheme_拒绝(self, policy):
        assert "禁止的 scheme" in policy.check_url("file:///etc/passwd")
        assert "禁止的 scheme" in policy.check_url("ftp://example.com")

    def test_缺_host_拒绝(self, policy):
        assert "缺少 host" in policy.check_url("http:///path")

    def test_localhost_字面量拒绝_不查_dns(self, policy):
        # localhost 在 DNS 解析前就该被拒（即便解析会失败）
        assert "localhost" in policy.check_url("http://localhost:8080/")
        assert "localhost" in policy.check_url("http://api.localhost/")

    def test_0000_拒绝(self, policy):
        assert "0.0.0.0" in policy.check_url("https://0.0.0.0/")

    def test_公网_ip_通过(self, policy, monkeypatch):
        monkeypatch.setattr(
            "astercli.web.network_policy.socket.getaddrinfo", fake_resolver(["93.184.216.34"])
        )
        assert policy.check_url("https://example.com/") is None

    def test_环回_ip_拒绝(self, policy, monkeypatch):
        monkeypatch.setattr(
            "astercli.web.network_policy.socket.getaddrinfo", fake_resolver(["127.0.0.1"])
        )
        assert "环回" in policy.check_url("http://some-host.com/")

    def test_内网_ip_拒绝(self, policy, monkeypatch):
        for private in ("10.0.0.5", "192.168.1.1", "172.16.0.9"):
            monkeypatch.setattr(
                "astercli.web.network_policy.socket.getaddrinfo", fake_resolver([private])
            )
            deny = policy.check_url("http://intranet.corp/")
            assert "内网" in deny

    def test_链路本地_拒绝(self, policy, monkeypatch):
        monkeypatch.setattr(
            "astercli.web.network_policy.socket.getaddrinfo", fake_resolver(["169.254.1.1"])
        )
        assert "链路本地" in policy.check_url("http://metadata.internal/")

    def test_dns_解析失败拒绝(self, policy, monkeypatch):
        def boom(host, *args, **kwargs):
            raise OSError("NXDOMAIN")

        monkeypatch.setattr("astercli.web.network_policy.socket.getaddrinfo", boom)
        assert "无法解析主机" in policy.check_url("https://nonexistent.invalid/")

    def test_多_ip_任一内网_全拒绝(self, policy, monkeypatch):
        # 双栈解析出公网 v6 + 内网 v4 → 整体拒绝（防混入内网 IP）
        monkeypatch.setattr(
            "astercli.web.network_policy.socket.getaddrinfo",
            fake_resolver(["2606:2800:220:1:248:1893:25c8:1946", "10.1.2.3"]),
        )
        assert "内网" in policy.check_url("https://dual-stack.com/")


class TestRateLimit:
    def test_窗口内超限被拒(self):
        p = NetworkPolicy(window_seconds=60, max_per_window=3)
        assert p.acquire() is None
        assert p.acquire() is None
        assert p.acquire() is None
        deny = p.acquire()
        assert deny is not None
        assert "请求过于频繁" in deny

    def test_提示包含剩余秒数(self):
        p = NetworkPolicy(window_seconds=60, max_per_window=1)
        p.acquire()
        deny = p.acquire()
        assert "秒后重试" in deny
