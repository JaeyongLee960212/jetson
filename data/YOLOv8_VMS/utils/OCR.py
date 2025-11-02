import requests
import uuid
import time
import json
import numpy as np
import socket

# Clova OCR API
OCR_KEY = 'ck1DYUFRclpRUlBWVnBlZmVwWGFQRmlhVlVSTENYWVY='
OCR_URL = 'https://anfmpduip9.apigw.ntruss.com/custom/v1/32335/01ae2140f9fc40e2379e144109beceae09e6946084f6b5f9912de323a121614b/general'

class OCR:
  
  def check_ip_connection(self, host='8.8.8.8', port=53, timeout=3):
        """
        Checks if the IP connection is available.
        Default host is Google Public DNS.
        """
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error as ex:
            return False
  
  def OCR(self,image_file):
    
    # if not self.check_ip_connection():
    #     print("No internet connection.")
    #     return None
    
    secret_key = OCR_KEY
    api_url = OCR_URL

    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
      ('file', open(image_file,'rb'))
    ]
    
    headers = {
      'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data = payload, files = files)

    response_json = response.json()

    texts = [i['inferText'] for i in response_json['images'][0]['fields']]
    
    if not texts:
      OCR_text = 'NONE'
    
    else:
      OCR_text = ' '.join(texts)

    return OCR_text