import socket

host = '192.0.0.1'
port = 5000

s = socket.socket()
s.bind((host,port))
s.listen(5)