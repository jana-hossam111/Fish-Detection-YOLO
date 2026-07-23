from ultralytics import YOLO

# تحميل أفضل موديل اتدرب
model = YOLO(r"C:\Users\Hadeel\Downloads\kb\runs\detect\Mort_Project\mort_detection\weights\best.pt")

# تجربة على الفيديو
results = model.predict(
    source=r"C:\Users\Hadeel\Downloads\test fish11.mp4",
    save=True,
    show=True,
    conf=0.25
)