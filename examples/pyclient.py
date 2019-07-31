import he_seal_client
import time

data = (1, 2, 3, 4)

hostname = 'localhost'
port = 34000
batch_size = 1

client = he_seal_client.HESealClient(hostname, port, batch_size, data, False)

while not client.is_done():
    time.sleep(1)

results = client.get_results()

print('results', results)
