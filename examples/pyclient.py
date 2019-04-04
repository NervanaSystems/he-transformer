import he_seal_client
import time

data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

hostname = 'localhost'
port = 34000
batch_size = 1

client = he_seal_client.HESealClient(hostname, port, batch_size, data)

while not client.is_done():
    time.sleep(1)

results = client.get_results()

print('results', results)
