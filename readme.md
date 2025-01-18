# Let Watermark Tell You the Truth

This is the code for the papar "Let Watermarks Speak: A Robust and Unforgeable Watermark for
Language Models".

## Requirements
```
gradio==5.10.0
numpy==2.2.1
torch==2.5.1
tqdm==4.66.5
transformers==4.46.3
```
For convenience, we provide a `requirements.txt` file. You can install all the requirements by running the following command: 
```
pip install -r requirements.txt
```
## Usage
To run the code, you need to first download the pre-trained language models. Then you can run the following command:
```
python main.py --m /path/to/model --server_name 0.0.0.0 --server_port 7860
```
The `--m` argument specifies the path to the pre-trained language model. The `--server_name` and `--server_port` arguments specify the IP address and port number of the server.
The code will start a Gradio server, which you can access by opening a web browser and navigating to the specified IP address and port number.

