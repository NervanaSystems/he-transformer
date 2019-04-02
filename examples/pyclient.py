import socket
import sys
import time

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('localhost', 34000)
print >> sys.stderr, 'connecting to %s port %s' % server_address
sock.connect(server_address)
'''header        | message_type | count        | data  |'''

msg = bytearray()
msg.append(15)  # For header

msg = bytes('99')

print(msg)

time.sleep(3)

sock.sendall(msg)

print('sent message')

time.sleep(5)