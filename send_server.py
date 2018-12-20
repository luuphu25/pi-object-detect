import aiohttp
import asyncio
import requests
import json
import time
import ujson
def create_info(object, addr):
	timestamp = time.time()
	normal_time = time.ctime()
	dic = { 'Time': normal_time, 'Object':object}
	push_json = json.dumps(dic)
	js = json.loads(push_json)
	#headers = "Content-Type: application/json"
        async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
                async with session.post(addr, json = data_js) as resp:
                        print(resp.status)
                        print(await resp.text())