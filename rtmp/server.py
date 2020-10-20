from rtmp.handshake import *
from rtmp.flv import *
from rtmp import connect
from queue import Queue
import threading, time


# https://github.com/pedroSG94/rtmp-rtsp-stream-client-java/blob/master/rtmp/src/main/java/com/github/faucamp/simplertmp/packets/Handshake.java
# 참고사이트 :java로 만들어진 rtmp_test client handshake


class Server:

    frame_queue = Queue()
    container = None

    def read(self):
        return self.frame_queue.get()

    def getFrameQueSize(self):
        return self.frame_queue.qsize()

    def start(self,PORT=1935):
        print("server start")
        time.perf_counter()

        self.PORT = PORT
        serverSock = socket(AF_INET, SOCK_STREAM)
        serverSock.bind(('', self.PORT))
        serverSock.listen(1)
        self.serverSock = serverSock
        self.clientSock, addr = self.serverSock.accept()

        # Handshake
        print("Start HandShaking")
        handshake = HandShake(self.clientSock)
        handshake.start()

        recvt = threading.Thread(target=self.receiveThMethod)
        recvt.daemon = True
        recvt.start()

    def receiveThMethod(self):
        # Connect
        print("\n\nStart Connecting")
        # conn = Command(self.clientSock)
        # onMetaData = conn.connect()
        cp = connect.Connect()
        cp.start(self.clientSock)
        onMetaData, chunkSize = cp.onMetaData, cp.chunkSize

        print("\n\nStart flv Decode",chunkSize)
        flv = FLV()
        flvframe_queue = Queue()

        flvframe_queue.put(flv.getHeaderBytes())
        # flvframe_queue.put(flv.RTMPtoFlvTag(conn.onMetaDataMessage))
        flvframe_queue.put(flv.RTMPtoFlvTag(onMetaData))


        t = threading.Thread(target=self.starDecode, args=(flvframe_queue, ))
        t.daemon = True
        t.start()

        msg = Message.readMessage(self.clientSock, False, chunkSize)

        flvtag = flv.RTMPtoFlvTag(msg)
        flvframe_queue.put(flvtag)

        while True:
            msg = Message.readMessage(self.clientSock,False, chunkSize)
            flvtag = flv.RTMPtoFlvTag(msg)
            flvframe_queue.put(flvtag)

    def starDecode(self,inbuffer: Queue):
        buf = QueIOBuffer(inbuffer)
        cc = av.open(buf)
        self.container = cc
        for s in cc.streams:
            s.options = {'tune':'zerolatency','genpts':'1'}

        packet = next(cc.demux())
        for packet in cc.demux():
            for frame in packet.decode():  # packet.decode시 list가 나오는데 가끔 빈 list가 나오는것을 무시
                self.frame_queue.put(frame)


class RTMPSendServer:
    frame_que = Queue()
    flvio = QueIOBuffer()
    chuncksize = 128
    starttime = None
    def write(self, decoded_frame : [av.AudioFrame, av.VideoFrame]):
        self.frame_que.put(decoded_frame)

    def start(self, width,height,PORT=8253):
        self.videowidth = width
        self.videoheight = height
        self.PORT = PORT

        et = threading.Thread(target=self.encodeThMethod)
        et.start()

        t=threading.Thread(target=self.writeThMethod)
        t.daemon = True
        t.start()

    def writeThMethod(self):

        serverSock = socket(AF_INET, SOCK_STREAM)
        serverSock.bind(('', self.PORT))
        serverSock.listen(1)
        self.serverSock = serverSock
        self.clientSock, addr = self.serverSock.accept()

        # Handshake
        print("Start HandShaking")
        handshake = HandShake(self.clientSock)
        handshake.start()
        # AMF0 Command connect('live’)
        msg = Message.readMessage(self.clientSock)

        # Window Acknowledgement Size 5000000
        Message.writeMessage(self.clientSock, b"\x02\x00\x00\x00\x00\x00\x04\x05\x00\x00\x00\x00\x00\x4c\x4b\x40",self.chuncksize)
        # Set Peer Bandwidth 5000000,Dynamic
        Message.writeMessage(self.clientSock, b"\x02\x00\x00\x00\x00\x00\x05\x06\x00\x00\x00\x00\x00\x4c\x4b\x40\x02",self.chuncksize)

        # Set Chunk Size 4000
        Message.writeMessage(self.clientSock,b"\x02\x00\x00\x00\x00\x00\x04\x01\x00\x00\x00\x00\x00\x00\x0f\xa0",self.chuncksize)
        # AMF0 Command _result('NetConnection.Connect.Success’)
        Message.writeMessage(self.clientSock, b"\x03\x00\x00\x00\x00\x00\xbe\x14\x00\x00\x00\x00\x02\x00\x07\x5f\x72\x65\x73\x75\x6c\x74\x00\x3f\xf0\x00\x00\x00\x00\x00\x00\x03\x00\x06\x66\x6d\x73\x56\x65\x72\x02\x00\x0d\x46\x4d\x53\x2f\x33\x2c\x30\x2c\x31\x2c\x31\x32\x33\x00\x0c\x63\x61\x70\x61\x62\x69\x6c\x69\x74\x69\x65\x73\x00\x40\x3f\x00\x00\x00\x00\x00\x00\x00\x00\x09\x03\x00\x05\x6c\x65\x76\x65\x6c\x02\x00\x06\x73\x74\x61\x74\x75\x73\x00\x04\x63\x6f\x64\x65\x02\x00\x1d\x4e\x65\x74\x43\x6f\x6e\x6e\x65\x63\x74\x69\x6f\x6e\x2e\x43\x6f\x6e\x6e\x65\x63\x74\x2e\x53\x75\x63\x63\x65\x73\x73\x00\x0b\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x02\x00\x15\x43\x6f\x6e\x6e\x65\x63\x74\x69\x6f\x6e\x20\x73\x75\x63\x63\x65\x65\x64\x65\x64\x2e\x00\x0e\x6f\x62\x6a\x65\x63\x74\x45\x6e\x63\x6f\x64\x69\x6e\x67\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09",self.chuncksize)
        # Window Acknowledgement Size
        msg.readMessage(self.clientSock)
        # AMF0 Command createStream()
        msg.readMessage(self.clientSock)


        # AMF0 Command _result()
        Message.writeMessage(self.clientSock, b"\x03\x00\x00\x00\x00\x00\x1d\x14\x00\x00\x00\x00\x02\x00\x07\x5f\x72\x65\x73\x75\x6c\x74\x00\x40\x00\x00\x00\x00\x00\x00\x00\x05\x00\x3f\xf0\x00\x00\x00\x00\x00\x00",self.chuncksize)
        # AMF0 Command getStreamLength()
        msg.readMessage(self.clientSock)

        # AMF0 Command play('bkild’)
        msg.readMessage(self.clientSock)
        # User Control Message Set Buffer Length 1,3000ms
        msg.readMessage(self.clientSock)

        # User Control Message Stream Begin 1
        Message.writeMessage(self.clientSock,b"\x02\x00\x00\x00\x00\x00\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01",self.chuncksize)

        # AMF0 Command onStatus('NetStream.Publish.Start’)
        Message.writeMessage(self.clientSock, b"\x05\x00\x00\x00\x00\x00\x60\x14\x01\x00\x00\x00\x02\x00\x08\x6f\x6e\x53\x74\x61\x74\x75\x73\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x00\x05\x6c\x65\x76\x65\x6c\x02\x00\x06\x73\x74\x61\x74\x75\x73\x00\x04\x63\x6f\x64\x65\x02\x00\x14\x4e\x65\x74\x53\x74\x72\x65\x61\x6d\x2e\x50\x6c\x61\x79\x2e\x53\x74\x61\x72\x74\x00\x0b\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x02\x00\x0a\x53\x74\x61\x72\x74\x20\x6c\x69\x76\x65\x00\x00\x09",self.chuncksize)
        # AMF0 Data |RtmpSampleAccess()
        Message.writeMessage(self.clientSock, b"\x05\x00\x00\x00\x00\x00\x18\x12\x01\x00\x00\x00\x02\x00\x11\x7c\x52\x74\x6d\x70\x53\x61\x6d\x70\x6c\x65\x41\x63\x63\x65\x73\x73\x01\x01\x01\x01",self.chuncksize)

        tes = threading.Thread(target=self.testMethod)
        tes.daemon = True
        tes.start()

        self.starttime = time.perf_counter()*1000

        flv = FLV()
        flv.readFlvHeader(self.flvio)
        rtmpmsg = flv.FlvTagtoRTMP(self.flvio) # 첫 tag는 onMetaData
        Message.writeMessage(self.clientSock, rtmpmsg, self.chuncksize)
        while True:
            rtmpmsg = flv.FlvTagtoRTMP(self.flvio)
            Message.writeMessage(self.clientSock,rtmpmsg,self.chuncksize)
    def testMethod(self):
            Message.readMessage(self.clientSock)
    def encodeThMethod(self):
        cc = av.open(self.flvio, 'w',format='flv')
        print("encode start")

        videoStream = cc.add_stream('h264')
        videoStream.options = {'tune':'zerolatency'}
        videoStream.width = self.videowidth
        videoStream.height = self.videoheight

        audioStream = cc.add_stream('aac')
        audioStream.options = {'tune':'zerolatency'}

        while True:
            frame = self.frame_que.get()
            if not self.starttime or frame.pts < self.starttime: continue # starttime이 None 이면 아직 obs와 연결이 안됬음을 의미
            self.encode(frame,cc,videoStream,audioStream)

    @staticmethod
    def encode(frame, cc, videoStream, audioStream):
        if isinstance(frame, av.VideoFrame):
            try:
                opacket = videoStream.encode(frame)
            except Exception as e:
                print('videoframe encode failed: ', e)

            if opacket is not None:
                try:
                    cc.mux(opacket)
                except Exception:
                    print('mux failed: ' + str(opacket))
        if isinstance(frame, av.AudioFrame):
            pts = frame.pts
            time_base = frame.time_base
            frame.pts = None
            try:
                opacket = audioStream.encode(frame)
            except Exception as e:
                print('audioframe encode failed: ', e)
            if opacket is not None:
                for p in opacket:
                    p.pts = pts
                    p.time_base = time_base

                try:
                    cc.mux(opacket)
                except Exception:
                    print('mux failed: ' + str(opacket))