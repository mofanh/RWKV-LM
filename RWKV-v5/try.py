import os

aa = os.listdir("/root/RWKV-LM/RWKV-v5/out")
print(aa)

list_p = []
for p in os.listdir("/root/RWKV-LM/RWKV-v5/out"):
    if p.startswith("rwkv") and p.endswith(".pth"):
        p = ((p.split("-"))[1].split("."))[0]
        if p != "final":
            if p == "init":
                p = -1
            else:
                p = int(p)
        print(f"01:{p}")
        list_p += [p]

list_p.sort()
max_p = list_p[-1]