# server end
`
docker run -it --rm -e https_proxy=${https_proxy} -p 9000:9000 -v "$PWD:/models/" sleepsontheflo/graphpipe-onnx:cpu --model=/models/3D-pose.onnx
`
note: model: input name: '0'; shape: 2x1070x17x2   
input.json: {"0": [1, [2, 1070, 34, 2]]}

# client end  

`python cilent.py`

