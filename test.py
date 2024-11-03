from ultralytics import YOLO

model = YOLO('pal-ai-model/palai_model.pt')
print(model.names)  # This should print a dictionary of class indices and their names