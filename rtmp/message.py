# [PyAV] Windows 설치기준 (다른 운영체제는 다름)

# ffmpeg [shared dev] 두가지 버전 다운로드
# dev 에서는 include, lib 폴더 복사
# shared 에서는 bin 폴더 복사
# include, lib, shared 폴더를 찾기 쉬운 경로의 폴더(예시 C:\ffmpeg )에 붙여넣기
# Win+Q -> [시스템 환경 변수 편집] 입력 -> [환경변수] -> xxx에 대한 사용자 변수 에서 [Path] 더블클릭
# [새로 만들기]를 입력후 위에서 만든 ffmpeg 폴더의 bin폴더 경로 입력  (예시 : C:\ffmpeg\bin )

# conda install av -c conda-forge
# 재부팅 하거나 프로그램을 껏다 켯음에도 pyav가 정상 작동안하면 아래 과정을통해 재설치

# PyAV git에서 zip 다운 받고 압축 푼 폴더로 이동해서
# pip install --upgrade -r tests/requirements.txt
# python setup.py build --ffmpeg-dir=C:\ffmpeg
# python setup.py install

# 에러가 뜰경우
# vc++14 build tools 필요할 가능성 높음
# https://visualstudio.microsoft.com/ko/vs/older-downloads/
# 재배포 가능 패키지 및 빌드 도구 -> Microsoft Build Tools 2015 업데이트 3  설치
# C:\Program Files (x86)\Windows Kits\8.1\bin\x86 에서 rc.exe & rcdll.dll복사
# C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin 에 붙여넣기

from socket import *
from rtmp import amf

# https://github.com/pedroSG94/rtmp-rtsp-stream-client-java/blob/master/rtmp/src/main/java/com/github/faucamp/simplertmp/packets/Handshake.java
# 참고사이트 :java로 만들어진 rtmp_test client handshake

SERVER_KEY = b'\x47\x65\x6e\x75\x69\x6e\x65\x20\x41\x64\x6f\x62\x65\x20\x46\x6c\x61\x73\x68\x20\x4d\x65\x64\x69\x61\x20\x53\x65\x72\x76\x65\x72\x20\x30\x30\x31\xf0\xee\xc2\x4a\x80\x68\xbe\xe8\x2e\x00\xd0\xd1\x02\x9e\x7e\x57\x6e\xec\x5d\x2d\x29\x80\x6f\xab\x93\xb8\xe6\x36\xcf\xeb\x31\xae'
FLASHPLAYER_KEY = b'\x47\x65\x6E\x75\x69\x6E\x65\x20\x41\x64\x6F\x62\x65\x20\x46\x6C\x61\x73\x68\x20\x50\x6C\x61\x79\x65\x72\x20\x30\x30\x31\xF0\xEE\xC2\x4A\x80\x68\xBE\xE8\x2E\x00\xD0\xD1\x02\x9E\x7E\x57\x6E\xEC\x5D\x2D\x29\x80\x6F\xAB\x93\xB8\xE6\x36\xCF\xEB\x31\xAE'


class MessageType:
    # Set Chunk Size, is used to notify the peer a newe maximum chunk size eto use
    SET_CHUNK_SIZE = b'\x01'

    # Abort Message, is used to notify the peer if it is waiting for chunks
    # to complete a message, then to discard the partially received message
    # over a chunk stream and abort processing of that message.
    ABORT = b'\x02'

    # * The client or the server sends the acknowledgment to the peer after
    # * receiving bytes equal to the window size. The window size is the
    # * maximum number of bytes that the sender sends without receiving
    # * acknowledgment from the receiver.
    ACKNOWLEDGEMENT = b'\x03'

    # * The client or the server sends this message to notify the peer about
    # * the user control events. This message carries Event type and Event
    # * data.
    # * Also known as a PING message in some RTMP implementations.
    USER_CONTROL_MESSAGE = b'\x04'

    # * The client or the server sends this message to inform the peer which
    # * window size to use when sending acknowledgment.
    # * Also known as ServerBW ("server bandwidth") in some RTMP implementations.
    WINDOW_ACKNOWLEDGEMENT_SIZE = b'\x05'

    # * The client or the server sends this message to update the output
    # * bandwidth of the peer. The output bandwidth value is the same as the
    # * window size for the peer.
    # * Also known as ClientBW ("client bandwidth") in some RTMP implementations.
    SET_PEER_BANDWIDTH = b'\x06'

    # * The client or the server sends this message to send audio data to the peer.
    AUDIO = b'\x08'

    # * The client or the server sends this message to send video data to the peer.
    VIDEO = b'\x09'

    # * The client or the server sends this message to send Metadata or any
    # * user data to the peer. Metadata includes details about the data (audio, video etc.)
    # * like creation time, duration, theme and so on.
    # * This is the AMF3-encoded version.
    DATA_AMF3 = b'\x0F'

    # * A shared object is a Flash object (a collection of name value pairs)
    # * that are in synchronization across multiple clients, instances, and
    # * so on.
    # * This is the AMF3 version: kMsgContainerEx=16 for AMF3.
    SHARED_OBJECT_AMF3 = b'\x10'

    # * Command messages carry the AMF-encoded commands between the client
    # * and the server.
    # * A command message consists of command name, transaction ID, and command object that
    # * contains related parameters.
    # * This is the AMF3-encoded version.
    COMMAND_AMF3 = b'\x11'

    # * The client or the server sends this message to send Metadata or any
    # * user data to the peer. Metadata includes details about the data (audio, video etc.)
    # * like creation time, duration, theme and so on.
    # * This is the AMF0-encoded version.
    DATA_AMF0 = b'\x12'

    # * A shared object is a Flash object (a collection of name value pairs)
    # * that are in synchronization across multiple clients, instances, and
    # * so on.
    # * This is the AMF0 version: kMsgContainer=19 for AMF0.
    SHARED_OBJECT_AMF0 = b'\x13'

    # * Command messages carry the AMF-encoded commands between the client
    # * and the server.
    # * A command message consists of command name, transaction ID, and command object that
    # * contains related parameters.
    # * This is the common AMF0 version, also known as INVOKE in some RTMP implementations.
    COMMAND_AMF0 = b'\x14'

    # * An aggregate message is a single message that contains a list of sub-messages.
    AGGREGATE_MESSAGE = b'\x16'


class ChunkType:
        # /** Full 12-byte RTMP chunk header */
        TYPE_0_FULL = [b"\x00", 12]

        # /** Relative 8-byte RTMP chunk header (message stream ID is not included) */
        TYPE_1_RELATIVE_LARGE = [b"\x01", 8]

        # /** Relative 4-byte RTMP chunk header (only timestamp delta) */
        TYPE_2_RELATIVE_TIMESTAMP_ONLY = [b"\x02", 4]

        # /** Relative 1-byte RTMP chunk header (no "real" header, just the 1-byte indicating chunk header type & chunk stream ID) */
        TYPE_3_RELATIVE_SINGLE_BYTE = [b"\x03", 1];


class Header:
    fmt = None # 2bit
    csid = 0 # 6bit
    timestamp = 0  # 3byte
    bodysize = None  # 3byte 기본값 : 128B
    type = None  # 1byte
    streamid = 0  # 4byte
    def __init__(self, chunksize = 128):
        None
    def getHeaderSize(self):
        if self.fmt == 0: size = 12
        elif self.fmt == 1: size = 8
        elif self.fmt == 2: size = 4
        elif self.fmt == 3: size = 1
        else: exit(1)

        size +=self.getCsidSize()

        return size

    def getCsidSize(self):  # 0이면 확장X
        if self.csid<64 : size = 0
        elif self.csid < 320: size = 1
        elif self.csid >= 320 : size = 2
        return size

class Message:
    header = None
    @classmethod
    def readMessage(cls, clientSock,isPrint:bool = True, chunksize=128, readSize = None):
        retmsg = Message()
        retmsg.header = Header()
        msg = None
        try:
            msg = clientSock.recv(1)
        except timeout:
            print("timeout : socket recv")
            return None

        retmsg.header.fmt = msg[0] >> 6
        retmsg.header.csid = msg[0] & 0b00111111
        if retmsg.header.csid == 0:
            msg += clientSock.recv(1)
            retmsg.header.csid = msg[1] + 64
        elif retmsg.header.csid == 0b111111:
            msg = clientSock.recv(2)
            retmsg.header.csid = msg[1] * 256 + msg[2] + 64

        readn = retmsg.header.getHeaderSize()-1
        msg = bytearray()
        while readn>0:
            msg.extend(clientSock.recv(readn))
            readn -= len(msg)

        # 명시되지 않은경우 이전 csid로 받은 data 재활용
        if retmsg.header.fmt == 3:
            exec('''retmsg.header.type = cls.cs%d_prevMsgType''' % (retmsg.header.csid))
            exec('''retmsg.header.bodysize = cls.cs%d_prevMsgBodySize''' % (retmsg.header.csid))
            exec('''retmsg.header.streamid = cls.cs%d_prevStreamID''' % (retmsg.header.csid))
            exec('''retmsg.header.timestamp = cls.cs%d_prevTimeStamp''' % (retmsg.header.csid))
        elif retmsg.header.fmt == 2:
            exec('''retmsg.header.type = cls.cs%d_prevMsgType''' % (retmsg.header.csid))
            exec('''retmsg.header.bodysize = cls.cs%d_prevMsgBodySize''' % (retmsg.header.csid))
            exec('''retmsg.header.streamid = cls.cs%d_prevStreamID''' % (retmsg.header.csid))

            retmsg.header.timestamp = msg[0:3]
            exec('''retmsg.header.timestamp = int.to_bytes(int.from_bytes(retmsg.header.timestamp,'big') + int.from_bytes(cls.cs%d_prevTimeStamp,'big'),3,'big') ''' % (retmsg.header.csid))

            exec('''cls.cs%d_prevTimeStamp = retmsg.header.timestamp''' % (retmsg.header.csid))
        elif retmsg.header.fmt == 1:
            exec('''retmsg.header.streamid = cls.cs%d_prevStreamID''' % (retmsg.header.csid))
            retmsg.header.timestamp = msg[0:3]

            retmsg.header.timestamp = msg[0:3]
            exec(
                '''retmsg.header.timestamp = int.to_bytes(int.from_bytes(retmsg.header.timestamp,'big') + int.from_bytes(cls.cs%d_prevTimeStamp,'big'),3,'big') ''' % (
                    retmsg.header.csid))

            retmsg.header.bodysize = msg[3:6]
            retmsg.header.type = msg[6:7]
            exec('''cls.cs%d_prevMsgType = retmsg.header.type''' % (retmsg.header.csid))
            exec('''cls.cs%d_prevMsgBodySize = retmsg.header.bodysize''' % (retmsg.header.csid))
            exec('''cls.cs%d_prevTimeStamp = retmsg.header.timestamp''' % (retmsg.header.csid))
        elif retmsg.header.fmt == 0:
            retmsg.header.timestamp = msg[0:3]
            retmsg.header.bodysize = msg[3:6]
            retmsg.header.type = msg[6:7]
            retmsg.header.streamid = msg[7:11] # streamid는 little-endian
            exec('''cls.cs%d_prevMsgType = retmsg.header.type''' % (retmsg.header.csid))
            exec('''cls.cs%d_prevMsgBodySize = retmsg.header.bodysize''' % (retmsg.header.csid))
            exec('''cls.cs%d_prevStreamID = retmsg.header.streamid''' % (retmsg.header.csid))
            exec('''cls.cs%d_prevTimeStamp = retmsg.header.timestamp''' % (retmsg.header.csid))

        # print(retmsg.header.__dict__)

        # http://osflash.org/_media/rtmp_spec.jpg
        # RTMP body는 128byte chunk들로 이루어짐
        # 각 chunk는 앞에 초기 헤더(fmt,csid,bodysize등으로 이루어진 것으로 생각됨) 또는 1 byte 헤더(0xC0+csid)
        # [Basic Header][128B][1B][128B][1B][128B].....
        # 128bytes [128] 129bytes[128][1][1]  등등
        # 만약 4096으로 chunk size로 설정했다면
        # [Basic Header][4096B][1B].....

        # Bytes : 1B 헤더 개수
        # 1~128 : 0
        # 129~256 : 1
        # 257~384 : 2
        retmsg.bodyList = []
        retmsg.body = bytearray()
        bodylen = int.from_bytes(retmsg.header.bodysize,'big')
        msgsize = bodylen + (bodylen-1)//chunksize # 1B 헤더 개수만큼 더읽음


        while msgsize>0:
            msg = clientSock.recv(msgsize)
            retmsg.body.extend(msg)
            msgsize -= len(msg)
        # print("Body", "".join("x%02x"%x for x in cls.body))
        for i in range(0,(bodylen-1)//chunksize): #set chunk size로 정한 바이트마다 1바이트 헤더제거 (따로 설정안했으면 128바이트)
            # 129 : 0,1
            try:
                del retmsg.body[(i+1)*chunksize]
            except Exception as e:
                print("len(body) : %d"%len(retmsg.body))
                print("(i+1)*chunksize : %d"%((i+1)*chunksize))
                print("bodylen : %d"%bodylen)
                print("bodylen + (bodylen-1)//chunksize : %d"%(bodylen + (bodylen-1)//chunksize))
                raise  e

        # print("Body", "".join("x%02x"%x for x in cls.body))
        if retmsg.header.fmt < 2 and 0x11<=retmsg.header.type[0]<=0x14:

            data= amf.BytesIO(retmsg.body)

            amfReader = amf.AMF0(data)
            while amfReader.data.eof()==False:
                st = amfReader.read()
                # print(st)
                retmsg.bodyList.append(st)
        elif retmsg.header.type[0] == 0x05: # Win Acknowledge Size
            retmsg.bodyList.append(int.from_bytes(retmsg.body, 'big'))
        elif retmsg.header.type[0] == 0x06: # Set Peer Bandwidth
            retmsg.bodyList.append(int.from_bytes(retmsg.body[0,4],'big')) # Window acknowledgement size
            retmsg.bodyList.append(retmsg.body[4]) # Limit type
        if isPrint: retmsg.print()
        return retmsg


    @staticmethod
    def writeMessage(clientSock:socket,data,chunksize=128):
        headerbytes =bytearray()
        fmt_csid=data[0:1]
        fmt = fmt_csid[0] >> 6
        csid = fmt_csid[0] & 0b00111111
        offset = 0
        if csid == 0: offset = 1
        elif csid == 0b111111: offset = 2
        headerbytes.extend(data[0:1+offset])

        rest = 12
        if fmt == 3: rest = 0
        elif fmt == 2: rest = 3
        elif fmt == 1: rest = 7
        elif fmt == 0: rest = 11

        headerbytes.extend(data[1+offset:12+offset])
        clientSock.send(headerbytes)

        writebodysize = len(data)-12-offset
        index = 12+offset
        while True:
            if writebodysize > chunksize:
                clientSock.send(data[index:index+chunksize]+bytes([0xC0+csid]))
                index+=chunksize
                writebodysize-=chunksize
            else:
                clientSock.send(data[index:])
                break

    def print(self):
        print(self.header.__dict__)
        for item in self.bodyList:
            if isinstance(item, amf.Object):
                keys = item._attrs.keys()
                for k in keys:
                    print(k, " : ", item._attrs.__getitem__(k))
            else: print(item)