"""Author: Nicola Vidovic"""

from typing import Callable

from connection import Connection
from zeroconf import IPVersion, ServiceBrowser, ServiceListener, Zeroconf


class NetworkBrowser():

    def __init__(self, service_name: str, connection_found_handler: Callable[[Connection], None]) -> None:
        self.zeroconf = Zeroconf()
        self.update_handler = UpdateHandler(connection_found_handler)
        self.browser = ServiceBrowser(
            self.zeroconf, service_name, self.update_handler
        )

    def close(self):
        self.zeroconf.close()
        self.browser.cancel()


class UpdateHandler(ServiceListener):

    def __init__(self, connection_found_handler: Callable[[Connection], None]) -> None:
        super().__init__()
        self.connection_found_handler = connection_found_handler

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        return

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        return

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        if self.connection_found_handler is None:
            return
        info = zc.get_service_info(type_, name)
        con = Connection(
            info.parsed_addresses(version=IPVersion.V4Only)[0],
            info.port
        )
        self.connection_found_handler(con)
        self.connection_found_handler = None


def dummy(con: Connection):
    print("con")


if __name__ == "__main__":
    nb = NetworkBrowser("_stoleAppNet._tcp.local.", dummy)
