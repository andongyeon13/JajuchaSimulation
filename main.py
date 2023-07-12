import pygame as p
import pygame.locals as pl
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np

from PIL import Image
from math import sin, cos, tan
from OpenGL.GL import shaders

import traceback
import time
import sys
import threading
import os
import socket
import cv2
import random

import FrameManager as fm
import Camera as c
import MyMath as mm
import Lights as l
import Actor as a
import glfunc as glf
import Jajucha as j
import jajucha.control as jc
from jajucha.graphics import Graphics


BLOCKY_TEXTURE = True

# 컨트롤러 클래스 (키보드 입력 및 마우스 입력 담당)
class Controller:
    # 생성자 함수
    def __init__(self, mainLoop, target):
        self.oldState = p.key.get_pressed()  # 이전에 눌린 키
        self.newState = p.key.get_pressed()  # 현재 눌린 키

        self.oldMouseState = p.mouse.get_pressed()  # 이전 마우스
        self.newMouseState = p.mouse.get_pressed()  # 현재 마우스

        self.target = target
        self.mainLoop = mainLoop

        self.mouseControl = False  # 마우스 컨트롤 사용 여부
        self.setMouseControl(self.mouseControl)  # 마우스 컨트롤 사용 여부 (생성자 함수 아래 참고)

        self.target.physics = True  # 물리법칙 적용 여부
        self.mainLoop.camera.pos_l = [41.5, 2.1, 0]  # 카메라 좌표
        self.mainLoop.camera.lookHorDeg = 0.0  # 카메라 수평 시선 각도
        self.mainLoop.camera.lookVerDeg = 0.0  # 카메라 수직 시선 각도
        self.target.renderFlag = False  # 렌더링 여부

    # 업데이트 함수
    def update(self, frameTerm):
        self.oldState = self.newState
        self.newState = p.key.get_pressed()

        self.oldMouseState = self.newMouseState
        self.newMouseState = p.mouse.get_pressed()

        if self.getStateChange(pl.K_ESCAPE) == 1:  # 시뮬레이션 종료
            p.quit()
            sys.exit(0)
        if self.getStateChange(pl.K_c) == 1:  # 마우스 사용 여부 전환
            if self.mouseControl:
                self.setMouseControl(False)
            else:
                self.setMouseControl(True)

        # 카메라 이동
        self.target.move(frameTerm, self.newState[pl.K_w], self.newState[pl.K_s], self.newState[pl.K_a], self.newState[pl.K_d], self.newState[pl.K_SPACE], self.newState[pl.K_LSHIFT])
        # 카메라 시야각 조절 (By 키보드)
        self.target.rotate(frameTerm, self.newState[pl.K_UP], self.newState[pl.K_DOWN], self.newState[pl.K_LEFT], self.newState[pl.K_RIGHT])

        # 카메라 시야각 조절 (By 마우스)
        if self.mouseControl:
            xMouse, yMouse = p.mouse.get_rel()
            if abs(xMouse) <= 1:
                xMouse = 0
            if abs(yMouse) <= 1:
                yMouse = 0
            self.target.rotateMouse(-xMouse, -yMouse)

    # 키 눌림 감지
    def getStateChange(self, index):
        if self.oldState[index] != self.newState[index]:
            return self.newState[index]
        else:
            return -1

    # 마우스 컨트롤 설정자
    def setMouseControl(self, boolean):
        if boolean:
            self.mouseControl = True
            p.event.set_grab(True)
            p.mouse.set_visible(False)
        else:
            self.mouseControl = False
            p.event.set_grab(False)
            p.mouse.set_visible(True)

# 텍스쳐 컨테이너 클래스
class TextureContainer:
    # 생성자 함수
    def __init__(self):
        self.data = { 0x37:self.getTexture("assets\\textures\\newnewtrack.png"),
                       0x317: self.getTexture("assets\\textures\\317.png")}

    # 텍스쳐 데이터 접근자
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError(item)

        return self.data[item]

    # 텍스쳐 접근자
    @staticmethod
    def getTexture(textureDir):
        print("\ttexture: ", textureDir)

        st2 = time.time()
        aImg = Image.open(textureDir)
        imgW = aImg.size[0]
        imgH = aImg.size[1]

        try:
            image_bytes = aImg.tobytes("raw", "RGBA", 0, -1)
            alpha = True
        except ValueError:
            image_bytes = aImg.tobytes("raw", "RGBX", 0, -1)
            alpha = False

        imgArray = np.array([x / 255 for x in image_bytes], np.float32)
        print("\t\tLoad textrue into np:", time.time() - st2)
        print("\t\tNdarray size:", imgArray.size*imgArray.itemsize/1024**2, "MB")

        st2 = time.time()
        textureId = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, textureId)
        gl.glTexStorage2D(gl.GL_TEXTURE_2D, 6, gl.GL_RGBA32F, imgW, imgH)

        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, imgW, imgH, gl.GL_RGBA, gl.GL_FLOAT, imgArray)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 6)

        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        # 픽셀 처리
        if not BLOCKY_TEXTURE:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        else:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        print("\t\trest: ", time.time() - st2)

        return textureId

# 정적 요소 클래스 (지형)
class StaticSurface:
    # 생성자 함수
    def __init__(self, vec1, vec2, vec3, vec4, surfaceVec, textureId, textureVerNum, textureHorNum, shininess, specularStrength):
        self.vertex1 = vec1
        self.vertex2 = vec2
        self.vertex3 = vec3
        self.vertex4 = vec4

        self.normal = surfaceVec
        self.textureHorNum = textureHorNum
        self.textureVerNum = textureVerNum
        self.shininess = shininess
        self.specularStrength = specularStrength

        self.textureId = textureId

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        vertices = np.array([*vec1.getXYZ(),
                             *vec2.getXYZ(),
                             *vec3.getXYZ(),
                             *vec1.getXYZ(),
                             *vec3.getXYZ(),
                             *vec4.getXYZ()], dtype=np.float32)
        size = vertices.size * vertices.itemsize

        self.verticesBuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.verticesBuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        del size, vertices

        textureCoords = np.array([0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1], dtype=np.float32)
        size = textureCoords.size * textureCoords.itemsize

        self.texCoordBuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.texCoordBuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, textureCoords, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        del size, textureCoords

    # 업데이트 함수
    def update(self):
        gl.glBindVertexArray(self.vao)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureId)


        gl.glUniform3f(2, *self.normal.getXYZ())
        gl.glUniform1f(3, self.textureHorNum)
        gl.glUniform1f(4, self.textureVerNum)


        gl.glUniform1f(11, self.shininess)
        gl.glUniform1f(53, self.specularStrength)

        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    # 그리기 함수
    def drawForShadow(self):
        gl.glBindVertexArray(self.vao)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureId)

        gl.glUniformMatrix4fv(3, 1, gl.GL_FALSE, mm.identityMat4())

        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    # 프로그램 연결 함수
    @staticmethod
    def getProgram() -> int:
        with open("shader_source\\2nd_vs.glsl") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log = glf.get_shader_log(vertexShader)
        if log:
            raise TypeError(log)

        with open("shader_source\\2nd_fs.glsl") as file:
            fragmentShader = shaders.compileShader(file.read(), gl.GL_FRAGMENT_SHADER)
        log = glf.get_shader_log(fragmentShader)
        if log:
            raise TypeError(log)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertexShader)
        gl.glAttachShader(program, fragmentShader)
        gl.glLinkProgram(program)

        print("Linking Log:", gl.glGetProgramiv(program, gl.GL_LINK_STATUS))

        gl.glDeleteShader(vertexShader)
        gl.glDeleteShader(fragmentShader)

        gl.glUseProgram(program)

        return program

# 지형 클래스
class Level:
    # 생성자 함수
    def __init__(self, mainLoop, depthMap):
        self.textureContainer = TextureContainer()
        self.mainLoop = mainLoop
        self.depthMap = depthMap

        self.ambientColor = (0.5, 0.5, 0.5)
        self.sunLightColor = (0.4, 0.4, 0.4)

        self.pointLights_l = [
        ]

        self.spotLights_l = [
            l.SpotLight((0, 0, 0), (1, 1, 1), 30, cos(mm.Angle(30).getRadian()), mm.Vec3(1, 0, 0))
        ]

        self.surfaceProgram = StaticSurface.getProgram()

        size = 50
        self.floor = StaticSurface(
            mm.Vec3(-size, 0, -size), mm.Vec3(-size, 0, size), mm.Vec3(size, 0, size), mm.Vec3(size, 0, -size),
            mm.Vec3(0, 1, 0),
            self.textureContainer[0x37],
            1, 1, 255, 0.2
        )
        self.wall1 = StaticSurface(
            mm.Vec3(-size, 37, -size), mm.Vec3(-size, 0, -size), mm.Vec3(size, 0, -size),
            mm.Vec3(size, 37, -size),
            mm.Vec3(0, 0, 1),
            self.textureContainer[0x317],
            1, 1, 8, 0.2
        )
        self.wall2 = StaticSurface(
            mm.Vec3(-size, 37, size), mm.Vec3(-size, 0, size), mm.Vec3(-size, 0, -size),
            mm.Vec3(-size, 37, -size),
            mm.Vec3(1, 0, 0),
            self.textureContainer[0x317],
            1, 1, 64, 0.2
        )
        self.wall3 = StaticSurface(
            mm.Vec3(size, 37, size), mm.Vec3(size, 0, size), mm.Vec3(-size, 0, size),
            mm.Vec3(-size, 37, size),
            mm.Vec3(0, 0, -1),
            self.textureContainer[0x317],
            1, 1, 8, 0.2
        )
        self.wall4 = StaticSurface(
            mm.Vec3(size, 37, -size), mm.Vec3(size, 0, -size), mm.Vec3(size, 0, size),
            mm.Vec3(size, 37, size),
            mm.Vec3(-1, 0, 0),
            self.textureContainer[0x317],
            1, 1, 64, 0.2
        )
        level = a.Actor()

        self.display = StaticSurfaceShadow(
            mm.Vec3(-2, 4, -5), mm.Vec3(-2, 0, -5), mm.Vec3(2, 0, -5), mm.Vec3(2, 4, -5),
            mm.Vec3(0, 0, 1), depthMap, 1, 1, 0, 0.0
        )

    # 업데이트 함수
    def update(self, term, projectMatrix, viewMatrix, camera, flashLight, shadowMat, sunLightDirection):

        self.sunLightColor = (0.5, 0.5, 0.5)
        gl.glClearBufferfv(gl.GL_COLOR, 0, (0.6, 0.6, 1.0, 1.0))

        self.ambientColor = (0.5, 0.5, 0.5)

        x, y, z = camera.getXYZ()
        y -= 0.5

        lightPos_t = tuple()
        lightColor_t = tuple()
        lightMaxDistance_t = tuple()
        lightCount_i = 0
        for x in self.pointLights_l:
            lightPos_t += x.getXYZ()
            lightColor_t += x.getRGB()
            lightMaxDistance_t += (x.maxDistance,)
            lightCount_i += 1

        gl.glUseProgram(self.display.program)

        gl.glUniformMatrix4fv(5, 1, gl.GL_FALSE, projectMatrix)
        gl.glUniformMatrix4fv(6, 1, gl.GL_FALSE, viewMatrix)
        gl.glUniformMatrix4fv(7, 1, gl.GL_FALSE, mm.identityMat4())

        gl.glUniform3f(8, *camera.getWorldXYZ())
        gl.glUniform3f(9, *self.ambientColor)

        gl.glUniform1i(10, lightCount_i)
        gl.glUniform3fv(12, lightCount_i, lightPos_t)
        gl.glUniform3fv(17, lightCount_i, lightColor_t)
        gl.glUniform1fv(22, lightCount_i, lightMaxDistance_t)

        gl.glUseProgram(self.surfaceProgram)

        gl.glUniformMatrix4fv(5, 1, gl.GL_FALSE, projectMatrix)
        gl.glUniformMatrix4fv(6, 1, gl.GL_FALSE, viewMatrix)
        gl.glUniformMatrix4fv(7, 1, gl.GL_FALSE, mm.identityMat4())

        gl.glUniformMatrix4fv(54, 1, gl.GL_FALSE, shadowMat)

        # gl.glUniform3f(8, *camera.parent.getXYZ())
        gl.glUniform3f(9, *self.ambientColor)

        gl.glUniform1i(10, lightCount_i)
        gl.glUniform3fv(12, lightCount_i, lightPos_t)
        gl.glUniform3fv(17, lightCount_i, lightColor_t)
        gl.glUniform1fv(22, lightCount_i, lightMaxDistance_t)

        gl.glUniform1i(55, 0)
        gl.glUniform1i(56, 1)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depthMap)

        gl.glUniform3f(gl.glGetUniformLocation(self.surfaceProgram, "sunLightColor"), *self.sunLightColor)
        gl.glUniform3f(gl.glGetUniformLocation(self.surfaceProgram, "sunLightDirection"), *sunLightDirection.getXYZ())

        self.floor.update()
        self.wall1.update()
        self.wall2.update()
        self.wall3.update()
        self.wall4.update()

    def drawForShadow(self, term):
        self.floor.drawForShadow()
        # self.loadedModelManager.drawForShadow()

# 그림자 처리
class StaticSurfaceShadow:
    # 생성자 함수
    def __init__(self, vec1, vec2, vec3, vec4, surfaceVec, textureId, textureVerNum, textureHorNum, shininess, specularStrength):
        self.vertex1 = vec1
        self.vertex2 = vec2
        self.vertex3 = vec3
        self.vertex4 = vec4

        self.normal = surfaceVec
        self.textureHorNum = textureHorNum
        self.textureVerNum = textureVerNum
        self.shininess = shininess
        self.specularStrength = specularStrength

        self.textureId = textureId
        self.program = self.getProgram()

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)


        vertices = np.array([*vec1.getXYZ(),
                             *vec2.getXYZ(),
                             *vec3.getXYZ(),
                             *vec1.getXYZ(),
                             *vec3.getXYZ(),
                             *vec4.getXYZ()], dtype=np.float32)
        size = vertices.size * vertices.itemsize

        self.verticesBuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.verticesBuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, vertices, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        del size, vertices

        textureCoords = np.array([0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1], dtype=np.float32)
        size = textureCoords.size * textureCoords.itemsize

        self.texCoordBuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.texCoordBuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, textureCoords, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        del size, textureCoords

    # 업데이트 함수
    def update(self):
        gl.glBindVertexArray(self.vao)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureId)

        gl.glUniform3f(2, *self.normal.getXYZ())
        gl.glUniform1f(3, self.textureHorNum_f)
        gl.glUniform1f(4, self.textureVerNum_f)

        gl.glUniform1f(11, self.shininess_f)
        gl.glUniform1f(53, self.specularStrength_f)

        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    # 그리기 함수
    def drawForShadow(self):
        gl.glBindVertexArray(self.vao)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureId)

        gl.glUniformMatrix4fv(3, 1, gl.GL_FALSE, mmath.identityMat4())

        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    # 프로그램 연결 함수
    @staticmethod
    def getProgram():
        with open("shader_source\\vs_shadow_draw.glsl") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log = glf.get_shader_log(vertexShader)
        if log:
            raise TypeError(log)

        with open("shader_source\\fs_shadow_draw.glsl") as file:
            fragmentShader = shaders.compileShader(file.read(), gl.GL_FRAGMENT_SHADER)
        log = glf.get_shader_log(fragmentShader)
        if log:
            raise TypeError(log)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertexShader)
        gl.glAttachShader(program, fragmentShader)
        gl.glLinkProgram(program)

        print("Linking Log:", gl.glGetProgramiv(program, gl.GL_LINK_STATUS))

        gl.glDeleteShader(vertexShader)
        gl.glDeleteShader(fragmentShader)

        gl.glUseProgram(program)

        return program

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
        self.generation = 0  # 세대 수
        self.unit = 0  # 개체 번호
        self.detected = 0  # 침범 감지 횟수
        self.runtime = 0  # 리워드 사이 주행 간격
        self.allRuntime = 0  # 개체 주행 시간

        self.shadowMap = glf.ShadowMap()
        self.level = Level(self, self.shadowMap.getTex())
        self.controller = Controller(self, self.camera)
        self.frameManager = fm.FrameManager(True)
        self.flashLight = True
        self.projectMatrix = None
        self.worldLoaded = False # 월드 로드 여부
        self.count = 0

        self.gene = [(2,2), (1,1), (2,1), (1,2)]  # 초기 유전자
        self.weight1 = self.gene[0][0]  # 가중치1 (속도 관여)
        self.weight2 = self.gene[0][1]  # 가중치2 (회전 관여)
        self.result = dict()  # 유전자별 리워드 값 저장

        self.rewardChecker = True
        self.reward = 0


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
        #image.save(os.path.join('subscreenshots', 'screenshot.png'))


    # 업데이트 함수
    def update(self):

        # 색상 추출
        #pixels = gl.glReadPixels(400, 400, 1, 1, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        #color = np.frombuffer(pixels, dtype=np.uint8)
        #print("Color at ({}, {}): {}, {}, {}".format(400, 400, color[0], color[1], color[2]))

        self.send_image()
        self.frameManager.update()

        if screenShot:
            self.screenshot()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        for event in p.event.get():
            if event.type == pl.QUIT:
                p.quit()
                sys.exit(0)
            elif event.type == pl.VIDEORESIZE:
                self.winSize = (event.dict['w'], event.dict['h'])
                self.onResize()

        self.controller.update(self.frameManager.getFrameTerm())

        driveCommand = self.send_command()

        if self.detected > self.unit:
            self.unit += 1

            if self.unit == len(self.gene):
                sorted_gene = sorted(self.result.items(), key=lambda x: x[1], reverse=True)
                print("generation: ", self.generation)
                print("best gene: ", sorted_gene[0][0], sorted_gene[1][0],"\n")
                new_gen = [sorted_gene[0][0], sorted_gene[1][0], (sorted_gene[0][0][0] + random.uniform(-0.1, 0.1), sorted_gene[0][0][1]+ random.uniform(-0.1, 0.1)), (sorted_gene[1][0][0] + random.uniform(-0.1, 0.1), sorted_gene[1][0][1] + random.uniform(-0.1, 0.1)), (sorted_gene[0][0][0]+ random.uniform(-0.1, 0.1), sorted_gene[1][0][1]+ random.uniform(-0.1, 0.1)), (sorted_gene[1][0][0]+ random.uniform(-0.1, 0.1), sorted_gene[0][0][1]+ random.uniform(-0.1, 0.1))]
                #print(self.result)
                #print(new_gen)

                self.gene.extend(new_gen)
                self.generation += 1

            self.weight1 = self.gene[self.unit][0]
            self.weight2 = self.gene[self.unit][1]
            self.runtime = 0
            self.allRuntime = 0


            print("Process: ", self.unit,"/", len(self.gene))

        self.runtime += self.frameManager.getFrameTerm()
        self.allRuntime += self.frameManager.getFrameTerm()

        if jc.Com:
            self.camera.Automove(self.frameManager.getFrameTerm()*self.weight1, int(driveCommand.split(" ")[2]))
            self.camera.Autorotate(self.frameManager.getFrameTerm()*self.weight2, int(driveCommand.split(" ")[1]))

        hor, ver = self.camera.getWorldDegree()

        if self.worldLoaded:
            lightProjection = mm.orthoMat4(-400.0, 400.0, -300.0, 300.0, -300.0, 300.0)
        else:
            lightProjection = mm.orthoMat4(-75.0, 75.0, -75.0, 75.0, -75.0, 75.0)

        sunLightDirection = mm.Vec4(0, 1, 0.5, 0).normalize()
        sunLightDirection = sunLightDirection.transform(mm.rotateMat4(time.time()*10%360, 0, 0, -1))

        # lightView = mmath.translateMat4(0, 10, 0, -1) * mmath.rotateMat4(0, 0, 1, 0) * mmath.rotateMat4(-30, 1, 0, 0)
        lightView = mm.getlookatMat4(sunLightDirection, mm.Vec4(0,0,0,0), mm.Vec4(0, 1, 0, 0))

        self.onResize()
        self.camera.pos_l[1] += 4
        viewMatrix = mm.translateMat4(*self.camera.getWorldXYZ(), -1) * mm.rotateMat4(0, 0, 1, 0) * mm.rotateMat4(-90, 1, 0, 0)
        self.level.update(self.frameManager.getFrameTerm(), self.projectMatrix, viewMatrix, self.camera,self.flashLight, lightView * lightProjection, sunLightDirection)
        pixels = gl.glReadPixels(290, 180, 90, 120, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        if list(set(pixels)).count(128) > 0:
            #print(self.unit)
            self.result[(self.weight1, self.weight2)] = self.reward
            #print(self.result)

            self.camera.pos_l = [41.5, 6.1, 0]
            self.camera.lookVerDeg = 0
            self.camera.lookHorDeg = 0

            print("line detected\nRuntime: ", self.allRuntime,"\nReward: ", self.reward, "\n")
            self.reward = 0
            self.allRuntime = 0
            self.detected += 1

        if list(set(pixels)).count(118) > 0 and self.rewardChecker:
            self.reward += 10 - self.runtime
            self.runtime = 0
            self.rewardChecker = False
        elif list(set(pixels)).count(118) == 0:
            self.rewardChecker = True

        self.camera.pos_l[1] -= 4

        ###### 리워드 확인 ######
        #print("reward: ", self.reward)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        viewMatrix = mm.translateMat4(*self.camera.getWorldXYZ(), -1) * mm.rotateMat4(hor, 0, 1, 0) * mm.rotateMat4(ver,1,0,0)
        self.level.update(self.frameManager.getFrameTerm(), self.projectMatrix, viewMatrix, self.camera, self.flashLight, lightView * lightProjection, sunLightDirection)


        if True:
            self.drawText((-0.95, 0.9, 0), "FPS : {}".format(self.frameManager.getFPS()[0]))
            self.drawText((-0.95, 0.8, 0), "Pos : {:.2f}, {:.2f}, {:.2f}".format(*self.camera.pos_l))
            self.drawText((-0.95, 0.7, 0), "Looking : {:.2f}, {:.2f}".format(self.camera.lookHorDeg, self.camera.lookVerDeg))

        p.display.flip()

    def send_command(self):
        HOST = '127.0.0.1'
        PORT = 12345

        driveRes = "command 0 0"
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))

        try:
            # 서버에 데이터 전송
            data = "givecommand".encode()
            client_socket.sendall(data)

            # 서버로부터 응답 수신
            driveRes = client_socket.recv(1024).decode()

            return driveRes
        except Exception as e:
            print('에러 발생:', e)

        # 클라이언트 소켓 닫기
        client_socket.close()

    def send_image(self):
        HOST = '127.0.0.1'
        PORT = 12345

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))

        try:
            # 이미지 캡처 및 인코딩
            img = self.screenshot()
            img = np.array(img)
            _, buffer = cv2.imencode('.jpg', img)
            img_data = buffer.tobytes()

            # 이미지 데이터 전송
            client_socket.sendall(img_data)
        except Exception as e:
            print('에러 발생:', e)

        # 클라이언트 소켓 닫기
        client_socket.close()

    # 화면 크기 조절 함수
    def onResize(self):
        w, h = self.winSize
        self.centerPos = (w / 2, h / 2)
        gl.glViewport(0, 0, w, h)
        self.projectMatrix = mm.perspectiveMat4(90.0, w / h, 0.1, 1000.0)
        #self.projectMatrix = mmath.orthoMat4(-10.0, 10.0, -10.0, 10.0, 0.0, 1000.0)

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

screenShot = False
def thread_1():
    global screenShot
    g = Graphics(j.Planning)  # 자주차 컨트롤러 실행
    g.root.mainloop()  # 클릭 이벤트 처리
    g.exit()  # 자주차 컨트롤러 종료

def thread_2():
    main()
    pass


if __name__ == '__main__':
    # 두 개의 스레드 생성
    thread1 = threading.Thread(target=thread_1)
    thread2 = threading.Thread(target=thread_2)

    # 스레드 시작
    thread1.start()
    thread2.start()

    # 모든 스레드의 실행이 종료될 때까지 기다림
    thread1.join()
    thread2.join()

    print("모든 스레드 실행 완료")
