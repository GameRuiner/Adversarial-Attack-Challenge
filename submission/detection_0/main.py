#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import uvicorn
from fastapi import FastAPI

from classifier import Classifier

classifier_obj = Classifier()

app = FastAPI()



@app.get("/detection/detect")
async def classify_image(image_path: str = ""):
    """ Extract Adversarial Decision Endpoint. """

    try:
        score, decision = classifier_obj.extract(image_path=image_path)
        report = {
            'image_path': image_path,
            'score': score,
            'decision': decision,
            'comment': 'OK'
        }

    except Exception as e:
        print("Error during classification:", e)
        report = {
            'image_path': image_path,
            'score': "ERROR",
            'decision': "ERROR",
            'comment': f"Error: {e}"
        }
    return report


if __name__ == '__main__':

    #to execute app in console
    uvicorn.run("main:app",
                host='0.0.0.0',
                port=7007)