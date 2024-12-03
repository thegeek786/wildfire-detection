from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov5mu.pt")

# Train the model using GPU
model.train(data=r"F:\woops_proj\Brain Tumor Detection\data.yaml", epochs=1, device="cuda")

# Set device to GPU
model.cuda()

# Perform inference on GPU
results = model(r"F:\woops_proj\Brain Tumor Detection\train\images\volume_1_slice_37_jpg.rf.54583c28079e35639d77fecca48785ae.jpg")
#print(results)