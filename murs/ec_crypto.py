from fastecdsa import keys, curve
from fastecdsa.point import Point
from Crypto import Random
import random
from Crypto.Cipher import AES
from time import time
from multiprocessing import Pool

randfile = Random.new()

my_curve = curve.P256
priv_key, pub_key = keys.gen_keypair(my_curve)


def elgamal_enc(pub_key, msg):
    k = random.getrandbits(255)
    c1 = curve.P256.G * k
    c2 = (pub_key * k).x + msg

    return c1, c2


def elgamal_dec(c1, c2, priv_key):
    solution = c2 - (c1 * priv_key).x

    return solution


def wrap_message(msg):
    Random.atfork()
    aes_key = randfile.read(32)
    aes_obj = AES.new(aes_key, AES.MODE_CBC, "0" * 16)
    enc_msg = aes_obj.encrypt(msg)
    aes_key_int = int.from_bytes(aes_key, byteorder="little")
    c0, c1 = elgamal_enc(pub_key, aes_key_int)

    blob = c0.x.to_bytes(35, byteorder="little", signed=False) \
           + c0.y.to_bytes(35, byteorder="little", signed=False) \
           + c1.to_bytes(42, byteorder="little", signed=False) \
           + enc_msg
    return blob


def unwrap_message(blob):
    c0x = int.from_bytes(blob[0:35], byteorder="little", signed=False)
    c0y = int.from_bytes(blob[35:70], byteorder="little", signed=False)
    c1 = int.from_bytes(blob[70:112], byteorder="little", signed=False)

    c0 = Point(c0x, c0y, curve=my_curve)
    aes_key = elgamal_dec(c0, c1, priv_key)
    aes_key_int = aes_key.to_bytes(32, byteorder="little", signed=False)

    enc_msg = blob[112:len(blob)]
    aes_obj = AES.new(aes_key_int, AES.MODE_CBC, "0" * 16)
    msg = aes_obj.decrypt(enc_msg)
    return msg


def enc_all(msg, num_layer):
    Random.atfork()
    blob = wrap_message(msg)
    for _ in range(num_layer - 1):
        new_blob = wrap_message(blob)
        blob = new_blob
    return blob


def peel_all(blob, num_layer):
    msg = unwrap_message(blob)
    for _ in range(num_layer - 1):
        new_msg = unwrap_message(msg)
        msg = new_msg
    return msg


def exp_enc_time():
    n = 1000
    num_layers = 10
    msgs = [randfile.read(32) for _ in range(n)]
    blobs = [randfile.read(32) for _ in range(n)]

    for num_layer in range(1, num_layers + 1):
        t0 = time()

        for i in range(n):
            blobs[i] = enc_all(msgs[i], num_layer)

        print(num_layer, time() - t0)


def exp_dec_time():
    n = 1000000
    num_layers = 10
    blobs = [randfile.read(32) for _ in range(n)]
    times = []
    for num_layer in range(1, num_layers + 1):

        blobs = p.map(wrap_message, blobs)

        t0 = time()

        recovered_msgs = p.map(unwrap_message, blobs)
        random.shuffle(recovered_msgs)

        t1 = time() - t0
        print(num_layer, t1)
        times.append(t1)

    print(times)


n_process = 32
p = Pool(n_process)

# exp_enc_time()
exp_dec_time()
