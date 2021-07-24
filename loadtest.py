#!/usr/bin/env python
# coding: utf-8
import pinecone
import pinecone.connector
import numpy as np
from multiprocessing import Process
import random

import time

pinecone.init('a307fee0-27d5-4cf8-99be-3aadd4ac517d', environment='alpha')  # alt
service = 'rohit-hindi-download'
d = 35

def gen_data_stream(d, max_id, n):
    for t in range(n):
        id_ = str(np.random.randint(0,max_id))
        vector = np.random.randn(d)/np.sqrt(d)
        yield (id_, vector)

def operation(worker_id, iterations=400):
    while(True):
        print("Starting process.")
        start = time.time()
        time_snap = start

        conn = pinecone.connector.connect(service)

        try:
            for _ in range(iterations):
                if time.time() - time_snap > 5000:
                    print(f"{worker_id} is still running.")
                    time_snap = time.time()

                #stream = gen_data_stream(d=35, max_id=1000000, n=100)
                #acks = conn.upsert(stream)
                data = list(gen_data_stream(d=35, max_id=1000000, n=100))
                for item in data:
                    ack = conn.unary_upsert(item)

        except Exception as e:
            print(f"{worker_id} saw exception {e}")


def run(service, iterations, processes=5):
    procs = []
    for i in range(processes):
        proc = Process(target=operation, args=(i, iterations))
        procs.append(proc)
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

if __name__ == '__main__':
    print("Starting...")
    iterations = 1000
    processes = 50
    s = time.time()
    run(service, iterations, processes=processes)
    print((iterations * processes) / (time.time() - s))

if __name__ == '__main__':
    pass