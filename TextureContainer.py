import pygame.locals as pl
import OpenGL.GL as gl

from PIL import Image

import numpy as np
import time

BLOCKY_TEXTURE = True

# 텍스쳐 컨테이너 클래스
class TextureContainer:
    # 생성자 함수
    def __init__(self):
        self.data = { 0x37:self.getTexture("assets\\textures\\newing.png"),
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