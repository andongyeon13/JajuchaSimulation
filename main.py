import pygame as p
import pygame.locals as pl
import OpenGL.GL as gl
import numpy as np

from OpenGL.GL import shaders
from PIL import Image
from math import sin, cos, tan

import socket
import pickle
import cv2
import traceback
import time
import sys
import os
import threading
import copy

import FrameManager as fm
import Camera as c
import MyMath as mm
import glfunc as glf
import Controller as con
import Level as le

S_frontImage = None
S_upImage = None
S_velocity = 40
S_steer = 0
S_reward = 0
S_done = False
S_reset = False
S_num = 0

G_deg = 0
G_distance = 0
G_pos = (41.5, 0)

Num = 0
SpawnPos_list = [[ 25.        , -56.175     , 90],
                 [-25.        , -40.125     ,270],
                 [-25.        ,  -8.025     , 90],
                 [-25.        ,56.175     , 270],
                 [41.66666667, 24.075, 180],
                 ]
RewardPos_list = [
    [41.66666667, 8.025],
    [ 41.66666667,  -8.025     ],
    [ 37.5       ,-24.075     ],
    [ 37.5       , -40.125     ],
    [ 37.5       , -52.1625    ],
    [ 25.        , -56.175     ], #
    [  8.33333333, -56.175     ],
    [ -8.33333333, -56.175     ],
    [-25.        , -56.175     ],
    [-37.5       , -52.1625    ],
    [-41.66666667, -40.125     ],
    [-25.        , -40.125     ], #
    [ -8.33333333, -40.125     ],
    [  4.16666667, -36.1125    ],
    [  8.33333333, -24.075     ],
    [  4.16666667, -12.0375    ],
    [ -8.33333333,  -8.025     ],
    [-25.        ,  -8.025     ], #
    [-41.66666667,  -8.025     ],
    [-41.66666667, 8.025     ],
    [-41.66666667,24.075     ],
    [-41.66666667,40.125     ],
    [-37.5       ,52.1625    ],
    [-25.        ,56.175     ],#
    [ -8.33333333,56.175     ],
    [  8.33333333,56.175     ],
    [ 25.        ,56.175     ],
    [ 37.5       ,52.1625    ],
    [ 41.66666667,40.125     ],
    [ 41.66666667,24.075     ],  #
    ]

# 메인 루프 클래스
class MainLoop:
    # 생성자 함수
    def __init__(self):
        p.init()
        p.font.init()
        self.winSize = (640, 480)
        self.centerPos = (self.winSize[0] / 2, self.winSize[1] / 2)
        self.screen = p.display.set_mode(self.winSize, pl.DOUBLEBUF | pl.OPENGL | pl.RESIZABLE) # 화면 해상도 초기화 (더블버퍼 모드, OpenGL 사용, 크기 조절 가능)
        p.display.set_caption("jajucha simulation")

        self.initGL()

        self.camera = c.Camera()

        self.shadowMap = glf.ShadowMap()
        self.level = le.Level(self, self.shadowMap.getTex())
        self.controller = con.Controller(self, self.camera)
        self.frameManager = fm.FrameManager(True)
        self.flashLight = True
        self.projectMatrix = None
        self.worldLoaded = False

        self.Velocity = 0
        self.Steer = 0
        self.driveCommand = "command 1 1"

    # OpenGL 초기화 함수
    @staticmethod
    def initGL():
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        pass

    # 스크린샷 함수
    def screenshot(self):
        # 프레임 버퍼와 픽셀 데이터를 저장할 메모리 할당
        pixels = gl.glReadPixels(0, 0, self.winSize[0], self.winSize[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        # OpenGL 좌표계와 이미지 좌표계를 일치시키기 위해 이미지를 상하 반전
        pixels = p.image.fromstring(pixels, (self.winSize[0], self.winSize[1]), 'RGB')
        pixels = p.transform.flip(pixels, False, True)

        # 이미지 저장
        image = Image.frombytes('RGB', (self.winSize[0], self.winSize[1]), p.image.tostring(pixels, 'RGB'))
        return image

    # 업데이트 함수
    def update(self):
        global S_frontImagem, S_upImage, S_velocity, S_steer, S_done, S_reset, S_num
        global G_distance, G_pos, G_deg, Num

        G_pos = [self.camera.pos_l[0], self.camera.pos_l[2]]

        if S_reset:
            self.driveCommand = "command 1 1"
            self.camera.pos_l = [SpawnPos_list[S_num//6][0], 2.1, SpawnPos_list[S_num//6][1]]
            self.camera.lookVerDeg = 0
            self.camera.lookHorDeg = SpawnPos_list[S_num//6][2]
            S_reset = False
            S_done = False
            S_velocity = 40
            S_steer = 0

        self.frameManager.update()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        for event in p.event.get():
            if event.type == pl.QUIT:
                p.quit()
                sys.exit(0)
            elif event.type == pl.VIDEORESIZE:
                self.winSize = (event.dict['w'], event.dict['h'])
                self.onResize()

        self.controller.update(self.frameManager.getFrameTerm())

        # 주행 처리
        if True:
            S_velocity = max(-20, S_velocity)
            S_velocity = min(100, S_velocity)
            S_steer = max(-50, S_steer)
            S_steer = min(50, S_steer)

            self.camera.Automove(self.frameManager.getFrameTerm()*S_velocity, int(self.driveCommand.split(" ")[2]))
            self.camera.Autorotate(self.frameManager.getFrameTerm()*S_steer, int(self.driveCommand.split(" ")[1]))

            G_deg = self.camera.lookHorDeg

        hor, ver = self.camera.getWorldDegree()

        if self.worldLoaded:
            lightProjection = mm.orthoMat4(-400.0, 400.0, -300.0, 300.0, -300.0, 300.0)
        else:
            lightProjection = mm.orthoMat4(-75.0, 75.0, -75.0, 75.0, -75.0, 75.0)
        sunLightDirection = mm.Vec4(0, 1, 0.5, 0).normalize()
        sunLightDirection = sunLightDirection.transform(mm.rotateMat4(time.time()*10%360, 0, 0, -1))
        lightView = mm.getlookatMat4(sunLightDirection, mm.Vec4(0,0,0,0), mm.Vec4(0, 1, 0, 0))

        self.onResize()

        # 시뮬레이션 화면 업데이트 (항공, 시뮬레이션 화면 상에는 적용되지 않음)
        self.camera.pos_l[1] += 4
        viewMatrix = mm.translateMat4(*self.camera.getWorldXYZ(), -1) * mm.rotateMat4(hor, 0, 1, 0) * mm.rotateMat4(-90, 1, 0, 0)
        self.level.update(self.frameManager.getFrameTerm(), self.projectMatrix, viewMatrix, self.camera,self.flashLight, lightView * lightProjection, sunLightDirection)

        # 주행 이미지 전달 (항공)
        pixels = gl.glReadPixels(275, 180, 90, 120, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        gray_pixels = gl.glReadPixels(0, 0, 640, 480, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image_array = np.frombuffer(gray_pixels, dtype=np.uint8).reshape((480, 640, 3))
        gray_array = np.dot(image_array[..., :3], [0.2989, 0.587, 0.114])
        S_upImage = np.where(gray_array > 0, 1, 0)

        # 주행 상황 처리
        if list(set(pixels)).count(128) > 0:
            S_done = True
            self.driveCommand = "command 0 0"
            print("line detected")
            Num = 0

        if Num == 29:
            S_done = True
            self.driveCommand = "command 0 0"
            print("Finished Cycle")
            Num = 0

        if abs(S_steer) >= 35 or S_velocity < -10:
            S_done = True
            self.driveCommand = "command 0 0"
            print('회전 또는 후진 심함')
            Num = 0

        self.camera.pos_l[1] -= 4

        # 시뮬레이션 화면 업데이트 (전방)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        viewMatrix = mm.translateMat4(*self.camera.getWorldXYZ(), -1) * mm.rotateMat4(hor, 0, 1, 0) * mm.rotateMat4(ver,1,0,0)
        self.level.update(self.frameManager.getFrameTerm(), self.projectMatrix, viewMatrix, self.camera, self.flashLight, lightView * lightProjection, sunLightDirection)

        # 주행 이미지 전달 (전방)
        img = np.array(self.screenshot())
        gray_img = np.dot(img[..., :3], [0.2989, 0.587, 0.114])
        gray_img = np.where(gray_img > 0, 1, 0)
        gray_img = gray_img[0:640][300:480]
        _, buffer = cv2.imencode('.jpg', gray_img)
        S_frontImage = buffer.tobytes()

        if True:
                self.drawText((-0.95, 0.9, 0), "FPS : {}".format(self.frameManager.getFPS()[0]))
                self.drawText((-0.95, 0.8, 0), "Pos : {:.2f}, {:.2f}, {:.2f}".format(*self.camera.pos_l))
                self.drawText((-0.95, 0.7, 0), "Looking : {:.2f}, {:.2f}".format(self.camera.lookHorDeg, self.camera.lookVerDeg))

        p.display.flip()

    # 화면 크기 조절 함수
    def onResize(self):
        w, h = self.winSize
        self.centerPos = (w / 2, h / 2)
        gl.glViewport(0, 0, w, h)
        self.projectMatrix = mm.perspectiveMat4(90.0, w / h, 0.1, 1000.0)

    # 화면 텍스트 출력 함수
    @staticmethod
    def drawText(position, textString):
        font = p.font.Font(None, 32)
        textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
        textData = p.image.tostring(textSurface, "RGBA", True)
        gl.glRasterPos3d(*position)
        gl.glDrawPixels(textSurface.get_width(), textSurface.get_height(), gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textData)

# main 함수
def main():
    try:
        mainLoop = MainLoop()
        while True:
            mainLoop.update()
            pass
    except SystemExit:
        pass
    except:
        p.quit()
        print('오류 발생')
        traceback.print_exc()
        time.sleep(1)
        input("Press any key to continue...")
        sys.exit(-1)

def reward_calculation(top_img, action):
    global S_velocity, S_steer, S_done, S_num
    global G_deg, G_distance, G_pos, Num

    G_deg = min(G_deg, 360-G_deg)

    v_action, s_action = action

    # need to check!!
    box = 240  # 박스의 중앙 y
    box_size = 60
    box_start = box - box_size
    box_end = box + box_size

    front = top_img[box_start][:]
    back = top_img[box_end][:]
    m = 320

    if len(np.where(front == 1)[0]) == 0 or len(np.where(back == 1)[0]) == 0:
        distance_mid = 0
        # gray_array = (top_img * 255).astype(np.uint8)
        # cv2.imwrite("gray_image.png", gray_array)
    else:
        fl, fr = np.where(front == 1)[0][0], np.where(front == 1)[0][-1]
        bl, br = np.where(back == 1)[0][0], np.where(back == 1)[0][-1]

        mf = (fr / 2 if fl >= 640 - fr else (fl + 640) / 2) if fr - fl < 100 else (fl + fr) / 2  # 100 is line 너비
        mb = (br / 2 if bl >= 640 - br else (bl + 640) / 2) if br - bl < 100 else (bl + br) / 2

        distance_mid = (abs(m - mf) + abs(m - mb)) / 10
    diff_vector = np.array(RewardPos_list[(S_num + Num + 1) % 30]) - np.array(G_pos)
    rad = np.arccos(np.dot(diff_vector, np.array([0, 1])) / np.linalg.norm(diff_vector))
    deg = rad * (180/np.pi)
    alpha = abs(G_deg - deg)
    reward = S_velocity * (np.cos(alpha) - np.sin(alpha) - distance_mid)
    print(np.cos(alpha),np.sin(alpha),distance_mid)
    # print(distance_mid)
    if distance_mid <= 20:
        reward *= 1.1

    if abs(G_pos[0] - RewardPos_list[(S_num + Num + 1) % 30][0]) < 5 and abs(
            G_pos[1] - RewardPos_list[(S_num + Num + 1) % 30][1]) < 5:
        if S_steer < 30:
            reward = 100 * max((30 - abs(S_steer)) / 25, 1)
        Num += 1
        print("mid goal reach: ", Num)

    # prevent wiggling
    if abs(s_action) > 0.7:
        reward *= 0.9

    # motivate accelaration
    if v_action > 0:
        reward *= 1.05

    if S_done:
        reward = - 200 * (abs(S_steer) / 20)

    return float(reward)

def handle_client(client_socket):
    global S_frontImage, S_upImage, S_velocity, S_steer, S_reward, S_done, S_reset, S_num

    # 클라이언트로부터 데이터 수신
    data = client_socket.recv(1024 * 1024)

    if data[0:9] == b'get_state':
        v_action, s_action = float(data.decode().split(" ")[1]), float(data.decode().split(" ")[2])
        S_velocity += float(v_action) * 10
        S_steer += float(s_action) * 1

        print("\n\n[Simulation State]\n\nVelocity: ", S_velocity,"\nSteer: ",S_steer,"\nReward: ",S_reward, "\nDone: ", S_done)

        S_reward = reward_calculation(S_upImage, (v_action, s_action))
        s_data = pickle.dumps([S_frontImage, S_velocity, S_steer, S_reward, S_done])
        client_socket.sendall(s_data)

    elif data[0:5] == b'reset':
        S_num = int(data.decode().split(" ")[1])
        print("\n\n[#####RESET#####]\n\n")
        S_reset = True
        client_socket.sendall(S_frontImage)

    # 클라이언트 소켓 닫기
    client_socket.close()

def server():
    while True:
        # 클라이언트의 연결 수락
        client_socket, addr = server_socket.accept()

        # 클라이언트 요청을 처리하는 스레드 생성
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == '__main__':
    # 서버의 IP 주소와 포트 번호
    HOST = '127.0.0.1'
    PORT = 12345

    # 소켓 생성 및 바인딩
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))

    # 클라이언트의 연결을 기다림
    server_socket.listen(10)  # 최대 소켓 수
    print('서버가 시작되었습니다.')

    # 두 개의 스레드 생성
    thread1 = threading.Thread(target=server)
    thread2 = threading.Thread(target=main)

    # 스레드 시작
    thread1.start()
    thread2.start()

    # 모든 스레드의 실행이 종료될 때까지 기다림
    thread1.join()
    thread2.join()

    print("모든 스레드 실행 완료")