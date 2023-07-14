import socket
import cv2
import numpy as np
import threading

# 서버의 IP 주소와 포트 번호
HOST = '127.0.0.1'
PORT = 12345

# 소켓 생성 및 바인딩
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))

# 클라이언트의 연결을 기다림
server_socket.listen(10)
print('서버가 시작되었습니다.')

img = None
img_data = None
command = b'command 0 0'

def handle_client(client_socket):
    global img, img_data, command

    try:
        # 클라이언트로부터 데이터 수신
        data = client_socket.recv(1024 * 100)

        if data == b'give':
            _, buffer = cv2.imencode('.jpg', img)
            img_data = buffer.tobytes()
            client_socket.sendall(img_data)
        elif data[0:7] == b'command':
            command = data
        elif data[0:11] == b'givecommand':
            client_socket.sendall(command)
        elif data:
            img_data = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.COLOR_BGR2RGB)
            #cv2.imwrite('received_image.jpg', img)
    except Exception as e:
        print('에러 발생:', e)

    # 클라이언트 소켓 닫기
    client_socket.close()

def main():
    while True:
        # 클라이언트의 연결 수락
        client_socket, addr = server_socket.accept()
        #print('클라이언트가 연결되었습니다:', addr)

        # 클라이언트 요청을 처리하는 스레드 생성
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == '__main__':
    main()
