import json
f = open('dataset/Laptop14/train.json', 'r')
content = f.read()
a = json.loads(content)

aspect_scope = []
cnt = 0
for item in a:
    aspect_scope.append(item)
    cnt += 1
print(cnt)

f.close()

f = open('dataset/Laptops_allennlp/train.json', 'r')
content = f.read()
b = json.loads(content)

aspect_scope_n = []
cnt = 0
for item in b:
    # print(item)
    for aspect in item['aspects']:
        item1 = {"token": aspect_scope[cnt]['token'], "postag": item['pos'], "edges":item['head'], "deprels": item['deprel'], "aspects": aspect_scope[cnt]['aspects'], "constituent": aspect_scope[cnt]['constituent'], "spans": aspect_scope[cnt]['spans'], "consadj":aspect_scope[cnt]['consadj']}
        cnt += 1
        aspect_scope_n.append(item1)

print(cnt)
f.close
# print(aspect_scope_n)

with open("dataset/Laptops_allennlp/train_scope.json","w") as f:
    json.dump(aspect_scope_n,f)
    print("加载入文件完成...")