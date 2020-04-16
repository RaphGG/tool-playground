import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.1.11", 6000))

time.sleep(3)
x = 0
while True:
  x+= 1
  print(x)
  s.sendall('click A\r\n'.encode())
  time.sleep(0.2)

  for y in range(5):
    s.sendall('click DLEFT\r\n'.encode())
    time.sleep(0.2)

  s.sendall('click DUP\r\n'.encode())
  time.sleep(0.2)

  for y in range(5):
    s.sendall('click DRIGHT\r\n'.encode())
    time.sleep(0.2)
  
  s.sendall('click A\r\n'.encode())
  time.sleep(0.2)