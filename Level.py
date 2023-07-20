import StaticSurface as ss
import StaticSurfaceShadow as sss
import TextureContainer as tc
import Lights as l
import MyMath as mm
import Actor as a

from math import sin, cos, tan

import OpenGL.GL as gl



# 지형 클래스
class Level:
    # 생성자 함수
    def __init__(self, mainLoop, depthMap):
        self.textureContainer = tc.TextureContainer()
        self.mainLoop = mainLoop
        self.depthMap = depthMap

        self.ambientColor = (0.5, 0.5, 0.5)
        self.sunLightColor = (0.4, 0.4, 0.4)

        self.pointLights_l = [
        ]

        self.spotLights_l = [
            l.SpotLight((0, 0, 0), (1, 1, 1), 30, cos(mm.Angle(30).getRadian()), mm.Vec3(1, 0, 0))
        ]

        self.surfaceProgram = ss.StaticSurface.getProgram()

        size = 50
        self.floor = ss.StaticSurface(
            mm.Vec3(-size, 0, -size * 1.284), mm.Vec3(-size, 0, size* 1.284), mm.Vec3(size, 0, size* 1.284), mm.Vec3(size, 0, -size* 1.284),
            mm.Vec3(0, 1, 0),
            self.textureContainer[0x37],
            1, 1, 255, 0.2
        )
        self.wall1 = ss.StaticSurface(
            mm.Vec3(-size, 37, -size* 1.284), mm.Vec3(-size, 0, -size* 1.284), mm.Vec3(size, 0, -size* 1.284),
            mm.Vec3(size, 37, -size* 1.284),
            mm.Vec3(0, 0, 1),
            self.textureContainer[0x317],
            1, 1, 8, 0.2
        )
        self.wall2 = ss.StaticSurface(
            mm.Vec3(-size, 37, size* 1.284), mm.Vec3(-size, 0, size* 1.284), mm.Vec3(-size, 0, -size* 1.284),
            mm.Vec3(-size, 37, -size* 1.284),
            mm.Vec3(1, 0, 0),
            self.textureContainer[0x317],
            1, 1, 64, 0.2
        )
        self.wall3 = ss.StaticSurface(
            mm.Vec3(size, 37, size* 1.284), mm.Vec3(size, 0, size* 1.284), mm.Vec3(-size, 0, size* 1.284),
            mm.Vec3(-size, 37, size* 1.284),
            mm.Vec3(0, 0, -1),
            self.textureContainer[0x317],
            1, 1, 8, 0.2
        )
        self.wall4 = ss.StaticSurface(
            mm.Vec3(size, 37, -size* 1.284), mm.Vec3(size, 0, -size* 1.284), mm.Vec3(size, 0, size* 1.284),
            mm.Vec3(size, 37, size* 1.284),
            mm.Vec3(-1, 0, 0),
            self.textureContainer[0x317],
            1, 1, 64, 0.2
        )
        level = a.Actor()

        self.display = sss.StaticSurfaceShadow(
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