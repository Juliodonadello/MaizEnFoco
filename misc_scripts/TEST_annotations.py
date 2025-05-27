import json

with open("DeepAgro\datasets/test_fffb7fa69/annotations.json", "r") as f:
    data = json.load(f)

anotaciones = data["annotations"]
images = data["images"]
categories = data["categories"]

errores = []

print
for ann in anotaciones:
    seg = ann.get("segmentation")
    if not isinstance(seg, list):
        errores.append((ann["id"], "Segmentation no tiene una sola lista interna"))
    elif len(seg[0]) % 2 != 0:
        errores.append((ann["id"], "Cantidad impar de coordenadas"))
    elif len(seg[0]) < 6:
        errores.append((ann["id"], "Menos de 3 puntos (polígono no válido)"))

print(f"Se encontraron {len(errores)} anotaciones con problemas:")
for eid, desc in errores:
    print(f"  - ID {eid}: {desc}")


print("Len images: ",len(images))
print("Len categories: ",len(categories))
print("Len annotations: ",len(anotaciones))
print("promedio instancias por imagen: ",len(anotaciones)/len(images))


imagen_1 = images[0]
categories_1 = categories[0]

