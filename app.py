#!/usr/bin/python
#-*-coding: utf-8 -*-
##from __future__ import absolute_import
######
import json
import time
from flask import Flask
from flask_restful import Resource, Api, reqparse
import requests
import predict_promotion as pp
import pandas as pd

app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return "Hello World!"

class detect_promotion(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('inputImagePath', type=str)
        dictp = parser.parse_args()
        kw = dictp['inputImagePath']
        result = pp.predictimg(kw)
        return {'result': result}

api.add_resource(detect_promotion, '/detect_promotion',endpoint='detect_promotion')

if __name__ == '__main__':
    app.run(threaded=True)
