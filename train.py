from ultralytics import YOLO

# Load a model
model = YOLO("YOLO-LiRa.yaml")  # build a new model from YAML
model = YOLO("YOLO-LiRa-AI-TOD.pt")  # load a pretrained model (recommended for training)
model = YOLO("YOLO-LiRa.yaml").load("YOLO-LiRa-AI-TOD.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="AI-TOD.yaml", epochs=200, imgsz=640)