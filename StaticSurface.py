import OpenGL.GL as gl
import numpy as np

from OpenGL.GL import shaders

import glfunc as glf

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

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    # 그리기 함수
    def drawForShadow(self):
        gl.glBindVertexArray(self.vao)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureId)

        gl.glUniformMatrix4fv(3, 1, gl.GL_FALSE, mm.identityMat4())

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    # 프로그램 연결 함수
    @staticmethod
    def getProgram():
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
