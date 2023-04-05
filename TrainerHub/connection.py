"""Author: Nicola Vidovic"""

import socket


class Connection:

    def __init__(self, host: str, port: int) -> None:
        self.remote_address = (host, port)
        self.socket = None
        # Stores data that was read but not returned, for the next read call.
        self._data_buffer = b""

    def start(self):
        self.socket = socket.create_connection(self.remote_address)

    def close(self):
        self.socket.close()

    def receive(self, bytes_to_receive: int) -> bytes:
        bytes_received = len(self._data_buffer)
        all_data_parts = [self._data_buffer]

        while bytes_received < bytes_to_receive:
            data_part = self.socket.recv(4096)
            if not data_part:
                print("Connection closed")
                return
            bytes_received += len(data_part)
            all_data_parts.append(data_part)

        all_data = b"".join(all_data_parts)
        self._data_buffer = all_data[bytes_to_receive:]
        return all_data[:bytes_to_receive]

    def send(self, data: bytes):
        self.socket.sendall(data)
