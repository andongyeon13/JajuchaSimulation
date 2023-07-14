from jajucha.planning import BasePlanning
from jajucha.graphics import Graphics
from jajucha.control import mtx
import cv2
import numpy as np
import time
import main
import threading
import socket

Velocity = 180

class Planning(BasePlanning):
    def __init__(self, graphics):
        super().__init__(graphics)
        # --------------------------- #
        self.vars.redCnt = 0
        self.vars.greenCnt = 0
        self.vars.stop = True
        self.vars.steer = 0
        self.vars.velocity = 0


    def process(self, t, frontImage, rearImage, frontLidar, rearLidar):
        global Velocity
        # Canny 이미지
        canny = self.canny(frontImage)
        self.imshow('canny', canny)

        # 차선 정보 파악
        V, L, R = self.gridFront(frontImage, cols=20, rows=10)

        # [주행 처리]
        if L[9] < 325: #- 10:
            e = 334 - L[9]
        elif R[9] < 316: #- 10:
            e = R[9] - 259
        else:
            e = 24

        steer = int(e / 2.5) + 2  # 계수 1/3, 조정 -6
        #print("[주행 정보]")
        #print ('L[0]=', L[0], 'L[8]=', L[8], 'L[9]=', L[9], end="  //  ")
        #print ('R[0]=', R[0], 'R[8]=', R[8], 'R[9]=', R[9])
        #print ('V[0]=', V[0], 'V[1]=', V[1], 'V[2]=', V[2],  'V[3]=', V[3], 'V[4]=', V[4], 'V[5]=', V[5], 'V[6]=', V[6], 'V[7]=', V[7], 'V[8]=', V[8], 'V[9]=', V[9])
        #print()

        self.vars.steer = steer
        self.vars.velocity = Velocity
        self.command()
        return self.vars.steer, self.vars.velocity

    def command(self):
        HOST = '127.0.0.1'
        PORT = 12345

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))

        try:
            # 서버에 데이터 전송
            data = ("command "+str(self.vars.steer)+" "+str(self.vars.velocity)).encode()
            client_socket.sendall(data)
        except Exception as e:
            print('에러 발생:', e)

        # 클라이언트 소켓 닫기
        client_socket.close()



if __name__ == "__main__":
    pass