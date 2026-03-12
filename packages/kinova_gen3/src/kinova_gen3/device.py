from __future__ import annotations

import collections
import collections.abc

if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.autogen.messages import Session_pb2


class DeviceConnection:
    TCP_PORT = 10000

    def __init__(self, *, ip: str, username: str, password: str) -> None:
        self._ip = ip
        self._username = username
        self._password = password
        self._transport = TCPTransport()
        self._router = RouterClient(self._transport, RouterClient.basicErrorCallback)
        self._session_manager: SessionManager | None = None

    def __enter__(self):
        self._transport.connect(self._ip, self.TCP_PORT)
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self._username
        session_info.password = self._password
        session_info.session_inactivity_timeout = 10_000
        session_info.connection_inactivity_timeout = 2_000

        self._session_manager = SessionManager(self._router)
        self._session_manager.CreateSession(session_info)
        return self._router

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._session_manager is not None:
            options = RouterClientSendOptions()
            options.timeout_ms = 1_000
            self._session_manager.CloseSession(options)
        self._transport.disconnect()


__all__ = ["DeviceConnection"]
