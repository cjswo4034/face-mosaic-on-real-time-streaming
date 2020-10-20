import threading
from queue import Queue

import time,sys
import cv2,av



class QueIOBuffer(object):

    def __init__(self,packet_queue=None):
        self.packet_queue = packet_queue if packet_queue!=None else Queue()
        self.buffer = b''
        self.open = True
        self.pos = 0


    def read(self, num):

        while self.open and len(self.buffer) < num:

            # print("GET PACKET")
            packet = self.packet_queue.get()
            if packet:
                self.buffer += packet
            else:
                print("QUEUE IS CLOSED")
                self.open = False

        ret = self.buffer[:num]
        self.buffer = self.buffer[num:]

        self.pos += len(ret)

        # print('read({}) -> {}'.format(num, len(ret)))
        return ret
    def write(self, data):
        self.packet_queue.put(data)
    def tell(self):
        print('tell() -> {}'.format(self.pos))
        return self.pos

    def xseek(self, pos, whence):
        raise NotImplementedError()



class FLV(object):

    previoustagsize = 0

    def getHeaderBytes(self):
       return b'FLV\x01\x05\x00\x00\x00\x09' # FLV 헤더


    def RTMPtoFlvTag(self, RTMP_Message):

        time = RTMP_Message.header.timestamp + b"\x00"

        flvtag = bytearray()
        flvtag.extend(int.to_bytes(self.previoustagsize, 4, 'big'))
        flvtag.extend(RTMP_Message.header.type)
        flvtag.extend(RTMP_Message.header.bodysize)
        flvtag.extend(time)
        flvtag.extend(b'\x00\x00\x00') # streamid
        flvtag.extend(RTMP_Message.body)

        self.previoustagsize += len(RTMP_Message.body) + 11
        return flvtag

    def readFlvHeader(self,myio):
        header = myio.read(9)
        if not header[0:3] == b'FLV': return False
        return header

    def FlvTagtoRTMP(self,myio):
        prevtagsize = myio.read(4) # 이전 tagsize는 필요없으니 읽고 버림
        RTMPbytes = bytearray()

        flvtag = myio.read(11)
        flvtype = flvtag[0:1] #1B
        datasize = flvtag[1:4] #3B
        timestamp = flvtag[4:7] #flv time4B ->rtmp_test time 3B 해당부분 수정필요
        streamid = flvtag[8:11] #3B
        if flvtype[0]==0x12: RTMPbytes.extend(b'\x04')#1B
        elif flvtype[0]==0x08: RTMPbytes.extend(b'\x07')#1B
        elif flvtype[0]==0x09: RTMPbytes.extend(b'\x06')#1B


        RTMPbytes.extend(timestamp)#3B
        RTMPbytes.extend(datasize) #3바이트
        RTMPbytes.extend(flvtype) #1B
        RTMPbytes.extend(int.to_bytes(1,4,'little')) #3B
        data = myio.read(int.from_bytes(datasize,'big'))
        RTMPbytes.extend(data)
        return RTMPbytes

