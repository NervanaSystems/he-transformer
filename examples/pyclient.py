import socket
import he_seal_client
import time

s = socket.socket()

data = (1, 2, 3, 4)

hostname = 'localhost'
port = 34000

client = he_seal_client.HESealClient(hostname, port, data)

while not client.is_done():
    time.sleep(1)

results = client.get_results()

print('results', results)
