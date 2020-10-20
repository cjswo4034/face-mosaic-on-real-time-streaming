
from rtmp.message import *
from rtmp import commands
from queue import Queue
import threading, time


#sender receiver 관리

#일단은 rtmp_test connect 과정에서만 작동하게끔 구상

class Connect:
    def __init__(self):
        self.clientSock = None
        self.rque = Queue()
        self.chunkSize = 128
        self.desiredChunkSize = 128
        self.onMetaData = None
        self.cond = threading.Condition()

    def start(self, clientSock, desiredChunkSize = 128):
        self.clientSock = clientSock
        self.desiredChunkSize = desiredChunkSize
        t = threading.Thread(target=self.receiveThread, args=(self.rque,))
        t.start()

        while True:
            msg = self.rque.get()

            if msg.header.type[0] == 0x14:
                command = msg.bodyList[0]
                cmd = commands
                if command == "connect":
                    self.cond.acquire()
                    cmd.connect(self.clientSock, desiredChunkSize)
                    # self.chunkSize = desiredChunkSize
                    self.cond.notifyAll()
                    self.cond.release()
                elif command == "createStream":
                    cmd.result(self.clientSock, msg.bodyList[1])
                elif command == "publish":
                    cmd.onStatus(self.clientSock,'NetStream.Publish.Start')

            elif msg.header.type[0] == 0x12:
                self.onMetaData = msg
                self.chunkSize = self.desiredChunkSize
                break

    def receiveThread(self, que: Queue):

        while True:

            self.cond.acquire()
            msg = Message.readMessage(self.clientSock, True, self.chunkSize)
            que.put(msg)

            if msg.header.type[0] == 0x01:  # set Chunck Size
                self.desiredChunkSize = self.chunkSize = int.from_bytes(msg.body, 'big')
            elif msg.header.type[0] == 0x12: # onMetaData
                self.starttime = time.perf_counter() * 1000
                break
            elif msg.header.type[0] == 0x14 and msg.bodyList[0] == "connect":
                self.cond.wait()

            self.cond.release()

