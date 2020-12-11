import logging

import azure.functions as func
from .nlpmodules.process import processText

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        logging.info(f'Request body: {req_body}')
    except:
        return func.HttpResponse('Failed to load json request')
    try:
        text = req_body.get('text')

        if text:
            outText = processText(text)
            return func.HttpResponse(f'\n {outText}')
        else:
            return func.HttpResponse(
                "Please pass a valid command in the request text"
            )
    except:
        return func.HttpResponse('Json in request was not in the correct format')
