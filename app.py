# an object of WSGI application 
from flask import Flask, request
model_name = 'distilbert-base-multilingual-cased'
class_names = ['hi','hospital','vaccine'] # from above
from transformers import AutoTokenizer
import numpy as np
tokenizer = AutoTokenizer.from_pretrained(model_name)
#from transformers import pipeline 
import os
app = Flask(__name__) # Flask constructor 
cf_port = os.getenv("PORT")

# create ONNX session
def create_onnx_session(onnx_model_path, provider='CPUExecutionProvider'):
    """
    Creates ONNX inference session from provided onnx_model_path
    """
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
    options = SessionOptions()
    options.intra_op_num_threads = 0
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend 
    session = InferenceSession(onnx_model_path, options, providers=[provider])
    session.disable_fallback()
    return session

pt_onnx_quantized_path='model-optimized-quantized.onnx'
sess = create_onnx_session(pt_onnx_quantized_path)


@app.route('/')	 
def hello():
    maxlen = 50
    print('hi')       
    # tokenize document and make prediction 
    tokens = tokenizer.encode_plus('hospital availability', max_length=50, truncation=True)
    tokens = {name: np.atleast_2d(value) for name, value in tokens.items()}
    print(sess.run(None, tokens)[0])
    print()
    print("predicted class: %s" % (class_names[np.argmax(sess.run(None, tokens)[0])]))
    return str(class_names[np.argmax(sess.run(None, tokens)[0])])

@app.route('/intent')	 
def hello2():
    data = request.args.get("data")
    print(data)
    maxlen = 50       
    # tokenize document and make prediction 
    tokens = tokenizer.encode_plus(data, max_length=50, truncation=True)
    tokens = {name: np.atleast_2d(value) for name, value in tokens.items()}
    print(sess.run(None, tokens)[0])
    print()
    print("predicted class: %s" % (class_names[np.argmax(sess.run(None, tokens)[0])]))
    return (str(class_names[np.argmax(sess.run(None, tokens)[0])]))


if __name__=='__main__':
    if cf_port is None:
        app.run(host = '0.0.0.0', port = 5000)
    else:
        app.run( host='0.0.0.0', port=int(cf_port))
