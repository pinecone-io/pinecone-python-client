"""Tests for TCP socket options: keep-alive and Nagle's algorithm."""

from __future__ import annotations

import socket
from unittest.mock import patch

from pinecone._internal.http_client import _build_socket_options


class TestBuildSocketOptions:
    def test_keepalive_enabled(self) -> None:
        opts = _build_socket_options()
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in opts

    def test_nagle_disabled(self) -> None:
        opts = _build_socket_options()
        assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in opts

    def test_linux_keepalive_params(self) -> None:
        with patch("pinecone._internal.http_client.sys") as mock_sys:
            mock_sys.platform = "linux"
            opts = _build_socket_options()
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 300) in opts
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60) in opts
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4) in opts

    def test_darwin_keepalive_params(self) -> None:
        with patch("pinecone._internal.http_client.sys") as mock_sys:
            mock_sys.platform = "darwin"
            opts = _build_socket_options()
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60) in opts
        # macOS should NOT have idle or count
        opt_names = [(level, name) for level, name, _ in opts]
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) not in opt_names
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPCNT) not in opt_names

    def test_windows_minimal_options(self) -> None:
        with patch("pinecone._internal.http_client.sys") as mock_sys:
            mock_sys.platform = "win32"
            opts = _build_socket_options()
        # Only keepalive enable and nodelay
        assert len(opts) == 2
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in opts
        assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in opts
