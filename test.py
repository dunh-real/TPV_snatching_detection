# from ollama import chat

# response = chat(
#     model = 'qwen2.5-coder:14b',
#     messages = [{'role': 'user', 'content': 'Write a python script to print numbers from 1 to 10'}],
# )
# print(response.message.content)

from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo26n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo26n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")

# train from pre-trained
model = YOLO("yolo26n.pt")  # pass any model type
results = model.train(epochs=5)

# val
# Load a YOLO model
model = YOLO("yolo26n.yaml")

# Train the model
model.train(data="coco8.yaml", epochs=5)

# Validate on training data
model.val()

