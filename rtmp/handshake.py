from rtmp.message import *
import sys, random, os, hmac, hashlib

class HandShake:
    PROTOCOL_VERSION = bytearray([0x03])
    HANDSHAKE_SIZE = 1536
    SHA256_DIGEST_SIZE = 32
    DIGEST_OFFSET_INDICATOR_POS = 8  # client라면 8+764
    KEY_OFFSET_INDICATOR_POS = 772

    def __init__(self,clientSock):
        self.clientSock = clientSock

    def start(self):
        self.readC0()
        self.readC1()
        self.writeS0()
        self.writeS1()
        self.writeS2()
        self.readC2()

    def readC0(self):
        c0 = self.clientSock.recv(1)
        if c0[0] != 0x03:
            print("not eq byte 0x03")
            sys.exit()
        print("{0:<20}{1:<5}{2:}".format("Success Read C0", ":", "".join("x%02x"%x for x in c0)))
        return c0

    def readC1(self):
        readn = 0
        c1 = bytearray()
        while (readn < self.HANDSHAKE_SIZE):
            tmp = self.clientSock.recv(self.HANDSHAKE_SIZE - readn)
            if (len(tmp) <= 0): exit()
            c1.extend(tmp)
            readn += len(tmp)

        if (len(c1) != self.HANDSHAKE_SIZE):
            print("C1 적절치 않은 패킷")
            sys.exit()
        self.c1 = c1
        print("{0:<20}{1:<5}{2:}".format("Success Read C1", ":", "".join("x%02x"%x for x in c1)))
        return c1

    def writeS0(self):
        self.clientSock.send(self.PROTOCOL_VERSION)
        print("{0:<20}{1:<5}{2:}".format("Success Write S0", ":", "".join("x%02x" % x for x in self.PROTOCOL_VERSION)))

    def writeS1(self):
        # offset>=0 764-4-32-offset>0 offset 728
        # ready [764 Key Block]
        keyOffset = random.randint(0, 764
                                   - 4
                                   - 128)
        partBeforePublicKey = os.urandom(keyOffset)
        partAfterPublicKey = os.urandom(764 - 4 - 128 - keyOffset)
        remaining = keyOffset
        keyOffsetBytes = bytearray(4)
        for i in range(0, 4):
            n = 3 - i
            if (remaining > 0xff):
                keyOffsetBytes[n] = 0xff
                remaining -= 0xff
            else:
                keyOffsetBytes[n] = remaining
                remaining -= remaining

        keyBlock = bytearray()
        keyBlock.extend(partBeforePublicKey)
        keys = self.generateKeyPair()
        keyBlock.extend(keys[0])
        keyBlock.extend(partAfterPublicKey)
        keyBlock.extend(keyOffsetBytes)
        # ready ~~~[764B Digest Block] ----------------[764B Key Block]

        digestOffset = random.randint(0, 764
                                      - 4
                                      - self.SHA256_DIGEST_SIZE)
        remaining = digestOffset
        digestOffsetBytes = bytearray(4)
        for i in range(0, 4):
            n = 3 - i
            if (remaining > 0xff):
                digestOffsetBytes[n] = 0xff
                remaining -= 0xff
            else:
                digestOffsetBytes[n] = remaining
                remaining -= remaining

        partBeforeDigest = os.urandom(digestOffset)  # digest Blockdptj 32B digest앞부분

        timeStamp = bytearray([0, 0, 0, 0])  # time.clock() 4B
        flashPlayerVer = bytearray([0, 0, 0, 0])  # bytearray([0x80, 0x00, 0x07, 0x02])#4B ver.11.2.202.233 해당값은 임의로 집어놓은 상태
        partAfterDigest = os.urandom(764
                                     - 4
                                     - self.SHA256_DIGEST_SIZE
                                     - digestOffset)

        partFront = bytearray(0)
        partFront.extend(timeStamp)  # 4B
        partFront.extend(flashPlayerVer)  # 4B
        partFront.extend(digestOffsetBytes)  # 4B
        partFront.extend(partBeforeDigest)  # aB

        tempArr = bytearray(0)
        tempArr.extend(partFront)
        tempArr.extend(partAfterDigest)  # bB a+b=728
        tempArr.extend(keyBlock)
        digest = self.calculateHash(tempArr, SERVER_KEY)

        s1 = bytearray(0)
        s1.extend(partFront)
        s1.extend(digest)
        s1.extend(partAfterDigest)
        s1.extend(keyBlock)
        self.clientSock.send(s1)
        print("{0:<20}{1:<5}{2:}".format("Success Write S1", ":", "".join("x%02x" % x for x in s1)))

    def writeS2(self):
        self.clientSock.send(self.c1)
        print("{0:<20}{1:<5}{2:}".format("Success Write S2", ":", "".join("x%02x" % x for x in self.c1)))

    def readC2(self):
        readn = 0
        c2 = bytearray()
        while (readn < self.HANDSHAKE_SIZE):
            tmp = self.clientSock.recv(self.HANDSHAKE_SIZE - readn)
            if (len(tmp) <= 0): exit()
            c2.extend(tmp)
            readn += len(tmp)

        if (len(c2) != self.HANDSHAKE_SIZE):
            print("C1 적절치 않은 패킷")
            sys.exit()
        print("{0:<20}{1:<5}{2:}".format("Success Read C2", ":", "".join("x%02x" % x for x in c2)))
        return c2

    @staticmethod
    def calculateHash(msg, key):  # Hmac-sha256
        return hmac.new(key, msg, hashlib.sha256).digest()

    @staticmethod
    def generateKeyPair():  # dummy key pair since we don't support encryption
        return (([int(random.randint(0, 255)) for i in range(128)], [0, 0, 0]))
