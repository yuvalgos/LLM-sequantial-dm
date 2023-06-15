from bardapi import Bard
import time
import requests

session = requests.Session()
bard = Bard(token="WwhdBQcgdlPF0h7QwMIWAJ370T7s4COan2K961L_wZi-yrdgl1grhWWAcPZJUHeJwAfJjg.", session=session)

t = time.time()
for i in range(10000):

    response = bard.get_answer(f"just checking your throughput, call number {i}")['content']
    if i % 500 == 0:
        print(response)
print(time.time() - t)