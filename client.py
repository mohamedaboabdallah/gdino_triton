import tritonclient.http as httpclient, numpy as np

cli = httpclient.InferenceServerClient(url="localhost:8000")

img = open("test.png","rb").read()
inputs = [
  httpclient.InferInput("image_bytes", [1], "BYTES"),
  httpclient.InferInput("text_prompt", [1], "BYTES")
]
inputs[0].set_data_from_numpy(np.array([img], dtype=object))
inputs[1].set_data_from_numpy(np.array(["find a person"], dtype=object))

outputs = [
  httpclient.InferRequestedOutput("boxes"),
  httpclient.InferRequestedOutput("phrases"),
  httpclient.InferRequestedOutput("annotated_image")
]

res = cli.infer("groundingdino_py", inputs=inputs, outputs=outputs)
print("boxes:", res.as_numpy("boxes"))
print("phrases:", res.as_numpy("phrases"))
